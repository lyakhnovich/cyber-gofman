from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _to_prompt(question: str, answer: str) -> str:
    return (
        "### Instruction:\n"
        f"{question.strip()}\n\n"
        "### Response:\n"
        f"{answer.strip()}"
    )


def _build_dataset(path: Path) -> Dataset:
    rows = _load_jsonl(path)
    texts = [_to_prompt(str(r.get("question", "")), str(r.get("answer", ""))) for r in rows]
    return Dataset.from_dict({"text": texts})


@dataclass
class TrainConfig:
    model_name: str
    train_file: Path
    val_file: Path
    out_dir: Path
    max_length: int
    lr: float
    epochs: int
    batch_size: int
    grad_accum: int


def train(cfg: TrainConfig) -> None:
    use_4bit = torch.cuda.is_available()
    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_cfg,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    train_ds = _build_dataset(cfg.train_file)
    val_ds = _build_dataset(cfg.val_file)

    def tok(batch: dict) -> dict:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=str(cfg.out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=max(1, cfg.batch_size),
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        bf16=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.model.save_pretrained(cfg.out_dir / "adapter")
    tokenizer.save_pretrained(cfg.out_dir / "adapter")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-file", default="app/data/datasets/igor_sft_train.jsonl")
    parser.add_argument("--val-file", default="app/data/datasets/igor_sft_val.jsonl")
    parser.add_argument("--out-dir", default="app/data/models/igor-lora")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model,
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        out_dir=Path(args.out_dir),
        max_length=args.max_length,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()
