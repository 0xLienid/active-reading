import os
import argparse
import torch
from typing import List, Literal
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import to_safe_model_name, gather_object_to_main
from dotenv import load_dotenv

load_dotenv()


def parse_strategies(outputs: List[str]) -> List[List[str]]:
    """
    Parse the strategies from the output.
    """
    return [[strategy.strip() for strategy in output.split("##") if strategy.strip()] for output in outputs]


def generate_strategies_task_agnostic(
    accelerator: Accelerator,
    model_name: str,
    per_device_batch_size: int,
    seed: int = 42
):
    """
    Generate task-agnostic data augmentation strategies.

    Args:
        accelerator: The accelerator to use.
        model_name: The name of the model to use.
        per_device_batch_size: The number of examples to process per device.
        seed: The seed to use for the random number generator.

    Returns:
        A dataset of with learning strategies for each example.
    """

    PROMPT = """
Consider the following document. What are some strategies specific to this document that I can use to help me learn 
and remember all of the information contained? Use markdown and prefix each strategy with ##

<document>
{document}
</document>
"""

    torch.manual_seed(seed)

    if accelerator.is_main_process:
        print(f"Loading model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print(f"Loading dataset and preparing dataloader")

    dataset = load_dataset("mfirth/simplewikiqa-pages", split="train").select(range(1))
    dataloader = DataLoader(
        dataset, batch_size=per_device_batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)

    all_local_rows = []
    for batch in tqdm(dataloader, desc=f"Generating on rank {accelerator.process_index}..."):
        batch_prompts = [PROMPT.format(document=page)
                         for page in batch["page"]]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                min_new_tokens=64,
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            outputs = tokenizer.batch_decode(
                outputs[:, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        strategies = parse_strategies(outputs)

        for i in range(len(outputs)):
            strategies_for_example = strategies[i]
            all_local_rows.append({
                "url": batch["url"][i],
                "title": batch["title"][i],
                "page": batch["page"][i],
                "strategies": strategies_for_example
            })

    accelerator.wait_for_everyone()
    gathered = gather_object_to_main(accelerator, all_local_rows)

    strategies_dataset = []
    if accelerator.is_main_process:
        for row in gathered:
            strategies_dataset.extend(row)

    return strategies_dataset


def generate_strategies_task_specific(
    accelerator: Accelerator,
    model_name: str,
    per_device_batch_size: int,
    task: Literal["trivia", "finance"],
    seed: int = 42
):
    """
    Generate task-specific data augmentation strategies.

    Args:
        model_name: The name of the model to use.
        per_device_batch_size: The number of examples to process per device.
        task: The task to generate strategies for.
        seed: The seed to use for the random number generator.
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--per-device-batch-size", type=int, required=True)
    parser.add_argument("--task-agnostic", action="store_true")
    parser.add_argument(
        "--task", type=Literal["trivia", "finance"], required=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator()

    if args.task_agnostic:
        strategies_dataset = generate_strategies_task_agnostic(
            accelerator, args.model_name, args.per_device_batch_size, args.seed)
    else:
        if args.task is None:
            raise ValueError("Task is required when task-agnostic is False")

        strategies_dataset = generate_strategies_task_specific(
            accelerator, args.model_name, args.per_device_batch_size, args.task, args.seed)

    if accelerator.is_main_process:
        strategies_dataset = Dataset.from_list(strategies_dataset)
        strategies_dataset.push_to_hub(f"mfirth/simplewikiqa-strategies-{to_safe_model_name(args.model_name)}", split="train",
                                       token=os.getenv("HF_TOKEN"))
