import os
import random
import logging
import json
import shutil
import torch
import wandb
from dataclasses import dataclass
from typing import List, Optional, Iterator, Callable
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from tqdm import tqdm
from validators.validator import Validator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str
    core_dataset: str
    core_dataset_split: str = "train"
    core_dataset_preprocess_function: Optional[Callable[[
        Dataset], Dataset]] = None
    core_dataset_generate_text_function: Optional[Callable] = None
    pretraining_dataset: Optional[str] = None
    pretraining_dataset_split: str = "train"
    pretraining_dataset_text_field: str = "text"
    pretraining_ratio: float = 0.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_steps: int = 10000
    max_seq_length: int = 2048
    warmup_steps: int = 100
    validation_steps: int = 500
    checkpoint_dir: str = "checkpoints"
    save_total_limit: int = 3
    seed: int = 42
    streaming: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


class MixedDatasetIterator(IterableDataset):
    """
    An iterable dataset that randomly samples from either the core training dataset
    or a pretraining corpus based on a configurable ratio.
    """

    def __init__(
        self,
        core_dataset: Dataset,
        pretraining_dataset: Optional[Dataset],
        pretraining_ratio: float,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
        core_generate_text_function: Optional[Callable] = None,
        pretraining_text_field: str = "text",
        seed: int = 42,
    ):
        self.core_dataset = core_dataset
        self.pretraining_dataset = pretraining_dataset
        self.pretraining_ratio = pretraining_ratio
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.core_generate_text_function = core_generate_text_function
        self.pretraining_text_field = pretraining_text_field
        self.seed = seed

        if pretraining_dataset is None and pretraining_ratio > 0:
            self.pretraining_ratio = 0.0
            self._warned_no_pretraining = True
        else:
            self._warned_no_pretraining = False

    def _tokenize(self, text: str) -> dict:
        return self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)
        core_iter = iter(self.core_dataset)
        pretraining_iter = (
            iter(self.pretraining_dataset) if self.pretraining_dataset else None
        )

        while True:
            use_pretraining = (
                pretraining_iter is not None
                and rng.random() < self.pretraining_ratio
            )

            try:
                if use_pretraining:
                    sample = next(pretraining_iter)
                    text = sample[self.pretraining_text_field]
                else:
                    sample = next(core_iter)
                    text = self.core_generate_text_function(sample)
            except StopIteration:
                if use_pretraining:
                    pretraining_iter = iter(self.pretraining_dataset)
                    sample = next(pretraining_iter)
                    text = sample[self.pretraining_text_field]
                else:
                    core_iter = iter(self.core_dataset)
                    sample = next(core_iter)
                    text = self.core_generate_text_function(sample)

            tokenized = self._tokenize(text)
            yield {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": tokenized["input_ids"].squeeze(0),
            }


def collate_fn(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    checkpoint_dir: str,
    step: int,
    validation_score: float,
    is_best: bool = False,
):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)

    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

    metadata = {
        "step": step,
        "validation_score": validation_score,
    }
    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best")
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        with open(os.path.join(best_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    logger.info(
        f"Saved checkpoint at step {step} with score {validation_score:.4f}")
    if is_best:
        logger.info(f"New best checkpoint!")


def run_validation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    validators: List[Validator],
    is_main_process: bool,
) -> dict:
    model.eval()
    results = {}

    for validator in validators:
        if is_main_process:
            logger.info(f"Running validation: {validator.name}")
        score = validator.validate(model, tokenizer)
        results[validator.name] = score
        if is_main_process:
            logger.info(f"  {validator.name}: {score:.4f}")

    model.train()
    return results


def train(
    config: TrainingConfig,
    validators: Optional[List[Validator]] = None,
):
    """
    Main training loop with mixed dataset sampling and validation.

    Args:
        config: Training configuration.
        validators: List of Validator instances for periodic evaluation.
    """
    if validators is None:
        validators = []

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb" if config.wandb_project else None,
    )

    if accelerator.is_main_process and config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "model_name": config.model_name,
                "core_dataset": config.core_dataset,
                "pretraining_dataset": config.pretraining_dataset,
                "pretraining_ratio": config.pretraining_ratio,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "max_steps": config.max_steps,
                "max_seq_length": config.max_seq_length,
                "warmup_steps": config.warmup_steps,
                "validation_steps": config.validation_steps,
                "seed": config.seed,
            },
        )

    if accelerator.is_main_process:
        logger.info(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        token=os.getenv("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, token=os.getenv("HF_TOKEN"))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        logger.info(f"Loading core dataset: {config.core_dataset}")

    core_streaming = config.streaming and config.core_dataset_preprocess_function is None
    core_dataset = load_dataset(
        config.core_dataset,
        split=config.core_dataset_split,
        streaming=core_streaming,
    )

    if config.core_dataset_preprocess_function is not None:
        if accelerator.is_main_process:
            logger.info("Applying core dataset preprocessing...")
        core_dataset = config.core_dataset_preprocess_function(core_dataset)

    pretraining_dataset = None
    if config.pretraining_dataset and config.pretraining_ratio > 0:
        if accelerator.is_main_process:
            logger.info(
                f"Loading pretraining dataset: {config.pretraining_dataset}")
        pretraining_dataset = load_dataset(
            config.pretraining_dataset,
            split=config.pretraining_dataset_split,
            streaming=config.streaming,
        )

    mixed_dataset = MixedDatasetIterator(
        core_dataset=core_dataset,
        pretraining_dataset=pretraining_dataset,
        pretraining_ratio=config.pretraining_ratio,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        core_generate_text_function=config.core_dataset_generate_text_function,
        pretraining_text_field=config.pretraining_dataset_text_field,
        seed=config.seed,
    )

    if mixed_dataset._warned_no_pretraining and accelerator.is_main_process:
        logger.warning(
            "pretraining_ratio > 0 but no pretraining_dataset provided. "
            "All samples will come from core_dataset."
        )

    dataloader = DataLoader(
        mixed_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    best_validation_score = float("-inf")
    global_step = 0
    total_loss = 0.0

    if accelerator.is_main_process:
        logger.info("Starting training...")
        logger.info(f"  Core dataset: {config.core_dataset}")
        logger.info(f"  Pretraining dataset: {config.pretraining_dataset}")
        logger.info(f"  Pretraining ratio: {config.pretraining_ratio}")
        logger.info(f"  Max steps: {config.max_steps}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(
            f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(
            f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    model.train()
    progress_bar = tqdm(
        total=config.max_steps,
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    for batch in dataloader:
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            total_loss += loss.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if global_step % config.validation_steps == 0:
            avg_loss = total_loss / config.validation_steps

            if accelerator.is_main_process:
                logger.info(f"Step {global_step}, Avg Loss: {avg_loss:.4f}")
                if config.wandb_project:
                    wandb.log({"train/loss": avg_loss,
                              "train/step": global_step})

            if validators:
                unwrapped_model = accelerator.unwrap_model(model)
                validation_results = run_validation(
                    unwrapped_model, tokenizer, validators, accelerator.is_main_process
                )

                avg_score = sum(validation_results.values()) / \
                    len(validation_results)
                is_best = avg_score > best_validation_score

                if is_best:
                    best_validation_score = avg_score

                if accelerator.is_main_process:
                    if config.wandb_project:
                        log_dict = {"val/step": global_step}
                        for name, score in validation_results.items():
                            log_dict[f"val/{name}"] = score
                        log_dict["val/avg_score"] = avg_score
                        if is_best:
                            log_dict["val/best_score"] = best_validation_score
                        wandb.log(log_dict)

                    save_checkpoint(
                        unwrapped_model,
                        tokenizer,
                        config.checkpoint_dir,
                        global_step,
                        avg_score,
                        is_best=is_best,
                    )
            else:
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_checkpoint(
                        unwrapped_model,
                        tokenizer,
                        config.checkpoint_dir,
                        global_step,
                        0.0,
                        is_best=False,
                    )

            total_loss = 0.0

        if global_step >= config.max_steps:
            break

    progress_bar.close()

    if accelerator.is_main_process:
        logger.info("Training complete!")
        final_path = os.path.join(config.checkpoint_dir, "final")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Saved final checkpoint to {final_path}")

        if config.wandb_project:
            wandb.finish()

    return model, tokenizer


if __name__ == "__main__":
    from validators.simplewikiqa_validator import SimpleWikiQAValidator

    def flatten_dataset(dataset: Dataset) -> Dataset:
        """Flatten the active reading dataset."""
        def expand_batch(batch: dict) -> dict:
            urls: List[str] = []
            titles: List[str] = []
            documents: List[str] = []
            strategies: List[str] = []
            applied: List[str] = []

            for url, title, doc, strats, applied_strats in zip(
                batch["url"],
                batch["title"],
                batch["document"],
                batch["strategy"],
                batch["applied_strategies"],
            ):
                if not strats or not applied_strats:
                    continue
                if len(strats) != len(applied_strats):
                    raise ValueError(
                        "strategy and applied_strategies lengths differ")

                for strategy, applied_strategy in zip(strats, applied_strats):
                    urls.append(url)
                    titles.append(title)
                    documents.append(doc)
                    strategies.append(strategy)
                    applied.append(applied_strategy)

            return {
                "url": urls,
                "title": titles,
                "document": documents,
                "strategy": strategies,
                "applied_strategy": applied,
            }

        return dataset.map(
            expand_batch,
            batched=True,
            remove_columns=list(dataset.column_names),
            desc="Flattening strategies and applied strategies",
        )

    def generate_text_function(sample: dict) -> str:
        """Convert a flattened active reading sample to training text."""
        return sample["applied_strategy"]

    NUM_GPUS = 4

    config = TrainingConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        core_dataset="mfirth/simplewikiqa-active-reading-meta-llama-Llama-3.1-8B-Instruct",
        core_dataset_preprocess_function=flatten_dataset,
        core_dataset_generate_text_function=generate_text_function,
        pretraining_dataset="mlfoundations/dclm-baseline-1.0",
        pretraining_ratio=0.1,
        batch_size=4,
        gradient_accumulation_steps=128 // (NUM_GPUS * 4),
        learning_rate=1e-5,
        max_seq_length=4096,
        max_steps=20000,
        validation_steps=2000,
        wandb_project="active-reading",
        wandb_run_name="llama-3.1-8b-finetune",
    )

    validators = [
        SimpleWikiQAValidator(batch_size=8),
    ]

    train(config, validators)
