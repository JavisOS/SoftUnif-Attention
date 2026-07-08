#!/usr/bin/env python3
"""Run a LoGiPT checkpoint with Logic-LM's released prompts/data/metric.

LoGiPT's public repository points to Logic-LM for evaluation scripts but does
not release a local HuggingFace generation entrypoint. This adapter keeps the
Logic-LM data format, few-shot prompt templates, and answer parsing, replacing
only the OpenAI API call with local `transformers` generation from a released
LoGiPT checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CHOICES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "A)",
    "B)",
    "C)",
    "D)",
    "E)",
    "F)",
    "G)",
    "H)",
    "A.",
    "B.",
    "C.",
    "D.",
    "E.",
    "F.",
    "G.",
    "H.",
]


def build_prompt(template: str, sample: dict[str, Any]) -> str:
    options = "\n".join(str(x).strip() for x in sample["options"])
    return (
        template.replace("[[CONTEXT]]", str(sample["context"]).strip())
        .replace("[[QUESTION]]", str(sample["question"]).strip())
        .replace("[[OPTIONS]]", options)
    )


def maybe_chat_wrap(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<s>[INST] {prompt} [/INST]"


def get_choice(answer_str: str) -> str | None:
    answer_str = answer_str.strip()
    for choice in CHOICES:
        if answer_str.startswith(choice):
            return choice.replace(")", "").replace(".", "")
    if answer_str.startswith(":"):
        return answer_str.replace(":", "").replace(".", "").strip()[:1] or None
    return None


def parse_answer(output: str) -> str | None:
    answer_str = output.strip()
    indicators = [
        "the correct option is",
        "the correct answer is",
        "The correct answer is",
        "The correct option is",
        "Thus, the answer is",
        "So, the correct option is",
    ]
    for indicator in indicators:
        idx = answer_str.rfind(indicator)
        if idx >= 0:
            answer_str = answer_str[idx + len(indicator) :].strip()
            break
    choice = get_choice(answer_str)
    if choice is not None:
        return choice
    match = re.search(r"\b([A-H])\s*[\).]", answer_str)
    return match.group(1) if match else None


def evaluate(results: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    parsed = 0
    for sample in results:
        pred = sample.get("prediction")
        gold = str(sample["answer"]).replace("(", "").replace(")", "").strip()
        parsed += int(pred is not None)
        correct += int(pred == gold)
    total = len(results)
    return {
        "accuracy": correct / max(total, 1),
        "parsed_ratio": parsed / max(total, 1),
        "correct": correct,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--logiclm-root", default="/root/TRUA/external_baselines/Logic-LM")
    parser.add_argument("--dataset-name", default="ProofWriter")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["Direct", "CoT"], default="CoT")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--keep-generations", action="store_true")
    args = parser.parse_args()

    root = Path(args.logiclm_root)
    data_path = root / "data" / args.dataset_name / f"{args.split}.json"
    template_path = root / "baselines" / "icl_examples" / f"{args.dataset_name}_{args.mode}.txt"
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[: args.limit]
    template = template_path.read_text(encoding="utf-8")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    results: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="LoGiPT Logic-LM eval"):
        batch = samples[start : start + args.batch_size]
        prompts = [maybe_chat_wrap(tokenizer, build_prompt(template, sample)) for sample in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for sample, text in zip(batch, texts):
            row = {
                "id": sample["id"],
                "answer": sample["answer"],
                "prediction": parse_answer(text),
                "predicted_answer": text.split("The correct option is:")[-1].strip(),
            }
            if args.keep_generations:
                row["generation"] = text
            results.append(row)

    metric = evaluate(results)
    payload = {
        "method": "LoGiPT",
        "model": args.model,
        "dataset": args.dataset_name,
        "split": args.split,
        "mode": args.mode,
        "source": "Logic-LM released data, prompt template, and option metric; local HF generation for LoGiPT checkpoint",
        "metric": metric,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
