import os
import argparse
import torch
from typing import Dict, List, Literal, Optional
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from utils import to_safe_model_name, gather_object_to_main
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MAX_PAGE_TOKENS = 1024


def parse_strategies(outputs: List[str]) -> List[List[str]]:
    """
    Parse the strategies from the output.
    """
    parsed: List[List[str]] = []
    for output in outputs:
        # 1) split by markdown "##" headings
        chunks = output.split("##")

        # 2) split each chunk into lines
        strategies: List[str] = []
        for chunk in chunks:
            for line in chunk.split("\n"):
                s = line.strip()
                if not s:
                    continue
                # Filter common prompt/document artifacts.
                if "</document>" in s or "</div>" in s:
                    continue
                # Filter out low-signal "strategies" that are likely headings,
                # numbering, or model noise.
                # - no spaces: single token like "Strategy:" or "1."
                # - < 4 words: too short to be an actionable strategy
                if " " not in s:
                    continue
                if len(s.split()) < 4:
                    continue
                strategies.append(s)

        # 3) de-dupe while preserving order
        seen = set()
        deduped: List[str] = []
        for s in strategies:
            if s in seen:
                continue
            seen.add(s)
            deduped.append(s)

        parsed.append(deduped)

    return parsed


def _chunk_text_by_tokens(text: str, tokenizer, max_tokens: int) -> List[str]:
    """
    Split a string into multiple strings such that each chunk is <= max_tokens
    when tokenized with `tokenizer` (using add_special_tokens=False).
    """
    if not text:
        return [""]

    input_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(input_ids) <= max_tokens:
        return [text]

    chunks: List[str] = []
    for start in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[start:start + max_tokens]
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append(chunk_text)

    # Fallback: if decoding produced nothing (rare), keep original text.
    return chunks or [text]


def preprocess_dataset_split_pages(
    dataset: Dataset,
    tokenizer,
    max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
    batch_size: int = 64,
    desc: Optional[str] = None,
) -> Dataset:
    """
    Expand a dataset by splitting each row into multiple rows based on the token
    length of the `page` column. Each resulting row has a `page` chunk with at
    most `max_page_tokens` tokens (tokenized with add_special_tokens=False).
    """
    if "page" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'page' column to split.")

    base_columns = list(dataset.column_names)

    def _split_batch(batch):
        out = {col: [] for col in base_columns}

        pages = batch["page"]
        for i in range(len(pages)):
            chunks = _chunk_text_by_tokens(
                pages[i], tokenizer, max_page_tokens)
            for ci, chunk in enumerate(chunks):
                for col in base_columns:
                    if col == "page":
                        out[col].append(chunk)
                    else:
                        out[col].append(batch[col][i])

        return out

    return dataset.map(
        _split_batch,
        batched=True,
        batch_size=batch_size,
        desc=desc or f"Splitting pages into <= {max_page_tokens} tokens",
    )


def process_batch(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_batch: Dict[str, torch.Tensor],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> List[List[str]]:
    with torch.no_grad():
        outputs = model.generate(
            tokenized_batch["input_ids"],
            attention_mask=tokenized_batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
        outputs = tokenizer.batch_decode(
            outputs[:, tokenized_batch["input_ids"].shape[1]:], skip_special_tokens=True)

    return parse_strategies(outputs)

def generate_strategies_task_agnostic(
    accelerator: Accelerator,
    model_name: str,
    per_device_batch_size: int,
    max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    if accelerator.is_main_process:
        print(f"Loading dataset and preparing dataloader")

    dataset = load_dataset("mfirth/simplewikiqa-pages", split="train")

    # Preprocess dataset by splitting long pages into <= max_page_tokens chunks.
    # To avoid duplicated work across distributed ranks, cache the expanded dataset
    # to disk (main process writes, others wait + read).
    cache_root = os.path.join(os.path.dirname(__file__), ".cache")
    split_cache_dir = os.path.join(
        cache_root,
        f"split_pages_{max_page_tokens}_tok_{to_safe_model_name(model_name)}",
    )
    done_file = os.path.join(split_cache_dir, "_DONE")

    if accelerator.is_main_process:
        os.makedirs(split_cache_dir, exist_ok=True)
        if not os.path.exists(done_file):
            expanded = preprocess_dataset_split_pages(
                dataset,
                tokenizer,
                max_page_tokens=max_page_tokens,
                desc=f"Preprocessing: split pages to <= {max_page_tokens} tokens",
            )
            expanded.save_to_disk(split_cache_dir)
            with open(done_file, "w") as f:
                f.write("ok\n")

    accelerator.wait_for_everyone()
    if os.path.exists(done_file):
        dataset = load_from_disk(split_cache_dir)
    else:
        # Fallback: if caching failed for some reason, proceed without caching.
        dataset = preprocess_dataset_split_pages(
            dataset,
            tokenizer,
            max_page_tokens=max_page_tokens,
            desc=f"Preprocessing (no-cache): split pages to <= {max_page_tokens} tokens",
        )

    dataloader = DataLoader(
        dataset, batch_size=per_device_batch_size)

    model, dataloader = accelerator.prepare(model, dataloader)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    all_local_rows = []
    to_reprocess = []
    for batch in tqdm(dataloader, desc=f"Generating on rank {accelerator.process_index}..."):
        batch_prompts = [PROMPT.format(document=page)
                         for page in batch["page"]]
        batch_inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True).to(accelerator.device)

        generated_strategies = process_batch(
            accelerator, unwrapped_model, tokenizer, batch_inputs, max_new_tokens=256, temperature=0.6, top_p=0.95)

        for i, generated_strategy_for_example in enumerate(generated_strategies):
            if len(generated_strategy_for_example) == 0:
                to_reprocess.append({
                    "url": batch["url"][i],
                    "title": batch["title"][i],
                    "page": batch["page"][i]
                })
                continue

            all_local_rows.append({
                "url": batch["url"][i],
                "title": batch["title"][i],
                "document": batch["page"][i],
                "strategies": generated_strategy_for_example
            })

    while len(to_reprocess) > 0:
        reprocessing_dataset = Dataset.from_list(to_reprocess)
        reprocessing_dataloader = DataLoader(
            reprocessing_dataset, batch_size=per_device_batch_size)
        
        to_reprocess = []
        for batch in tqdm(reprocessing_dataloader, desc=f"Reprocessing on rank {accelerator.process_index}..."):
            batch_prompts = [PROMPT.format(document=page)
                             for page in batch["page"]]
            batch_inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True).to(accelerator.device)

            generated_strategies = process_batch(
                accelerator, unwrapped_model, tokenizer, batch_inputs, max_new_tokens=256, temperature=0.6, top_p=0.95)

            for i, generated_strategy_for_example in enumerate(generated_strategies):
                if len(generated_strategy_for_example) == 0:
                    to_reprocess.append({
                        "url": batch["url"][i],
                        "title": batch["title"][i],
                        "page": batch["page"][i]
                    })
                    continue

                all_local_rows.append({
                    "url": batch["url"][i],
                    "title": batch["title"][i],
                    "document": batch["page"][i],
                    "strategies": generated_strategy_for_example
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
    max_page_tokens: int = DEFAULT_MAX_PAGE_TOKENS,
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
    parser.add_argument("--max-page-tokens", type=int,
                        default=DEFAULT_MAX_PAGE_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator()

    if args.task_agnostic:
        strategies_dataset = generate_strategies_task_agnostic(
            accelerator, args.model_name, args.per_device_batch_size, args.max_page_tokens, args.seed)
    else:
        if args.task is None:
            raise ValueError("Task is required when task-agnostic is False")

        strategies_dataset = generate_strategies_task_specific(
            accelerator, args.model_name, args.per_device_batch_size, args.task, args.max_page_tokens, args.seed)

    if accelerator.is_main_process:
        strategies_dataset = Dataset.from_list(strategies_dataset)
        strategies_dataset.push_to_hub(f"mfirth/simplewikiqa-strategies-{to_safe_model_name(args.model_name)}-test", split="train",
                                       token=os.getenv("HF_TOKEN"))
