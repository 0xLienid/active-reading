import os
import argparse
import torch
from typing import List, Literal, Dict, Any, Tuple, Optional
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import to_safe_model_name, gather_object_to_main
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAX_RETRIES = 5


def flatten_dataset(dataset: Dataset) -> Dataset:
    """
    Flatten strategies from each row so that each row has a single strategy, meaning that
    each row expands into n_strategies rows.
    """
    def _expand_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        urls: List[Any] = []
        titles: List[Any] = []
        documents: List[Any] = []
        strategies: List[Any] = []

        for url, title, doc, strats in zip(
            batch["url"], batch["title"], batch["document"], batch["strategies"]
        ):
            if not strats:
                continue

            for strategy in strats:
                urls.append(url)
                titles.append(title)
                documents.append(doc)
                strategies.append(strategy)

        return {
            "url": urls,
            "title": titles,
            "document": documents,
            "strategy": strategies,
        }

    return dataset.map(
        _expand_batch,
        batched=True,
        remove_columns=list(dataset.column_names),
        desc="Flattening strategies",
    )


def unflatten_dataset(dataset: Dataset | List[Dict[str, Any]]) -> Dataset:
    """
    Group rows by url, title, and page, and concatenate the strategies and applied strategies into a single row.
    """

    if not isinstance(dataset, Dataset):
        dataset = Dataset.from_list(dataset)

    doc_col = "document" if "document" in dataset.column_names else "page"
    key_cols = ("url", "title", doc_col)

    rename_cols = {
        "applied_strategy": "applied_strategies",
    }

    value_cols = [c for c in dataset.column_names if c not in key_cols]

    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in dataset:
        key = (row["url"], row["title"], row[doc_col])
        if key not in grouped:
            grouped[key] = {"url": key[0], "title": key[1], doc_col: key[2]}
            for c in value_cols:
                grouped[key][rename_cols.get(c, c)] = []

        for c in value_cols:
            grouped[key][rename_cols.get(c, c)].append(row.get(c))

    return Dataset.from_list(list(grouped.values()))


def push_checkpoint(
    accelerator: Accelerator,
    all_local_rows: List[Dict[str, Any]],
    repo_id: Optional[str],
    step: int,
):
    """
    Gather current rows from all ranks and push a checkpoint to the Hub.
    """
    if not repo_id:
        return

    accelerator.wait_for_everyone()
    gathered = gather_object_to_main(accelerator, all_local_rows)

    if not accelerator.is_main_process:
        return

    combined: List[Dict[str, Any]] = []
    for rows in gathered:
        combined.extend(rows)

    if not combined:
        return

    ds = unflatten_dataset(combined)
    ds.push_to_hub(
        repo_id,
        split="train",
        token=os.getenv("HF_TOKEN"),
        commit_message=f"checkpoint step {step}",
    )


def generate_active_reading_dataset(
    accelerator: Accelerator,
    model_name: str,
    dataset_name: str,
    per_device_batch_size: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    seed: int = 42,
    checkpoint_steps: int = 0,
    repo_id: Optional[str] = None,
):
    """
    Generate an active reading dataset.

    Args:
        accelerator: The accelerator to use.
        model_name: The name of the model to use.
        per_device_batch_size: The number of examples to process per device.
        seed: The seed to use for the random number generator.
        checkpoint_steps: If > 0, periodically push checkpoints every N steps.
        repo_id: Optional HF repo id to push checkpoints to.

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print(f"Loading dataset and preparing dataloader")

    dataset = load_dataset(dataset_name, split="train")
    dataset = flatten_dataset(dataset)
    dataloader = DataLoader(
        dataset, batch_size=per_device_batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    step = 0
    all_local_rows = []
    to_reprocess = []
    for batch in tqdm(dataloader, desc=f"Generating on rank {accelerator.process_index}..."):
        batch_prompts = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT.format(
                strategy=strategy, document=page)}
        ] for strategy, page in zip(batch["strategy"], batch["document"])]
        batch_inputs = tokenizer.apply_chat_template(
            batch_prompts, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True, padding=True).to(model.device)

        with torch.no_grad():
            outputs = unwrapped_model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id
            )
            outputs = tokenizer.batch_decode(
                outputs[:, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        for i, output in enumerate(outputs):
            if len(output) == 0 or len(output.split()) < 30:
                to_reprocess.append({
                    "url": batch["url"][i],
                    "title": batch["title"][i],
                    "document": batch["document"][i],
                    "strategy": batch["strategy"][i],
                })
                continue

            all_local_rows.append({
                "url": batch["url"][i],
                "title": batch["title"][i],
                "document": batch["document"][i],
                "strategy": batch["strategy"][i],
                "applied_strategy": outputs[i]
            })

        step += 1
        if checkpoint_steps > 0 and step % checkpoint_steps == 0:
            push_checkpoint(accelerator, all_local_rows, repo_id, step)

    num_retries = 0
    while len(to_reprocess) > 0 and num_retries < max_retries:
        reprocessing_dataset = Dataset.from_list(to_reprocess)
        reprocessing_dataloader = DataLoader(
            reprocessing_dataset, batch_size=per_device_batch_size)

        to_reprocess = []
        for batch in tqdm(reprocessing_dataloader, desc=f"Reprocessing on rank {accelerator.process_index}..."):
            batch_prompts = [[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT.format(
                    strategy=strategy, document=page)}
            ] for strategy, page in zip(batch["strategy"], batch["document"])]
            batch_inputs = tokenizer.apply_chat_template(
                batch_prompts, add_generation_prompt=True, return_tensors="pt", return_in_dict=True, padding=True).to(model.device)

            with torch.no_grad():
                outputs = unwrapped_model.generate(
                    batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_new_tokens=2048,
                    pad_token_id=tokenizer.eos_token_id
                )
                outputs = tokenizer.batch_decode(
                    outputs.sequences[:, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            for i, output in enumerate(outputs):
                if len(output) == 0 or len(output.split()) < 30:
                    to_reprocess.append({
                        "url": batch["url"][i],
                        "title": batch["title"][i],
                        "document": batch["document"][i],
                        "strategy": batch["strategy"][i],
                    })
                    continue

                all_local_rows.append({
                    "url": batch["url"][i],
                    "title": batch["title"][i],
                    "document": batch["document"][i],
                    "strategy": batch["strategy"][i],
                    "applied_strategy": outputs[i]
                })

            step += 1
            if checkpoint_steps > 0 and step % checkpoint_steps == 0:
                push_checkpoint(accelerator, all_local_rows, repo_id, step)

        num_retries += 1

    accelerator.wait_for_everyone()
    gathered = gather_object_to_main(accelerator, all_local_rows)

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
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    args = parser.parse_args()

    accelerator = Accelerator()
    repo_id = f"mfirth/simplewikiqa-active-reading-{to_safe_model_name(args.model_name)}"
    active_reading_dataset = generate_active_reading_dataset(
        accelerator,
        args.model_name,
        args.dataset_name,
        args.per_device_batch_size,
        args.max_retries,
        args.seed,
        args.checkpoint_steps,
        repo_id,
    )

    if accelerator.is_main_process:
        print(f"Saving active reading dataset")
        active_reading_dataset.push_to_hub(
            repo_id,
            split="train",
            token=os.getenv("HF_TOKEN"),
        )
