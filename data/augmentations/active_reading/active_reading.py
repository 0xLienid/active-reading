import os
import argparse
import torch
from typing import List, Literal, Dict, Any, Tuple
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import to_safe_model_name
from dotenv import load_dotenv

load_dotenv()


def flatten_dataset(dataset: Dataset) -> Dataset:
    """
    Flatten strategies from each row so that each row has a single strategy, meaning that
    each row expands into n_strategies rows.
    """
    return dataset.map(lambda x: [{"url": x["url"], "title": x["title"], "page": x["page"], "strategy": strategy} for strategy in x["strategy"]])


def unflatten_dataset(dataset: Dataset) -> Dataset:
    """
    Group rows by url, title, and page, and concatenate the strategies and applied strategies into a single row.
    """

    key_cols = ("url", "title", "page")
    value_cols = [c for c in dataset.column_names if c not in key_cols]

    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in dataset:
        key = (row["url"], row["title"], row["page"])
        if key not in grouped:
            grouped[key] = {"url": key[0], "title": key[1], "page": key[2]}
            for c in value_cols:
                grouped[key][c] = []

        for c in value_cols:
            grouped[key][c].append(row.get(c))

    return Dataset.from_list(list(grouped.values()))


def generate_active_reading_dataset(
    accelerator: Accelerator,
    model_name: str,
    dataset_name: str,
    per_device_batch_size: int,
    seed: int = 42
):
    """
    Generate an active reading dataset.

    Args:
        accelerator: The accelerator to use.
        model_name: The name of the model to use.
        per_device_batch_size: The number of examples to process per device.
        seed: The seed to use for the random number generator.

    Returns:
        A dataset applying learning strategies to documents.
    """

    PROMPT = """
Here's a learning strategy:
{strategy}

Apply this strategy to the following document:
<document>
{document}
</document>
"""

    torch.manual_seed(seed)

    if accelerator.is_main_process:
        print(f"Loading model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if accelerator.is_main_process:
        print(f"Loading dataset and preparing dataloader")

    dataset = load_dataset(dataset_name, split="train")
    dataset = flatten_dataset(dataset)
    dataloader = DataLoader(
        dataset, batch_size=per_device_batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)

    all_local_rows = []
    for batch in tqdm(dataloader, desc=f"Generating on rank {accelerator.process_index}..."):
        batch_prompts = [PROMPT.format(strategy=strategy, document=page)
                         for strategy, page in zip(batch["strategy"], batch["page"])]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", return_dict=True,
                                 padding=True, truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id
            )
            outputs = tokenizer.batch_decode(
                outputs.sequences[:, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        for i in range(len(outputs)):
            all_local_rows.append({
                "url": batch["url"][i],
                "title": batch["title"][i],
                "page": batch["page"][i],
                "strategy": batch["strategy"][i],
                "applied_strategy": outputs[i]
            })

    accelerator.wait_for_everyone()
    gathered = accelerator.utils.gather_object(all_local_rows)

    active_reading_dataset = []
    if accelerator.is_main_process:
        for row in gathered:
            active_reading_dataset.extend(row)

    active_reading_dataset = unflatten_dataset(active_reading_dataset)
    return active_reading_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator()
    active_reading_dataset = generate_active_reading_dataset(
        accelerator, args.model_name, args.dataset_name, args.per_device_batch_size, args.seed)

    if accelerator.is_main_process:
        print(f"Saving active reading dataset")
        active_reading_dataset.push_to_hub(f"mfirth/simplewikiqa-active-reading-{to_safe_model_name(args.model_name)}", split="train",
                                           token=os.getenv("HF_TOKEN"))
