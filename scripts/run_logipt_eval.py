#!/usr/bin/env python3
"""Evaluate a LoGiPT causal-LM checkpoint on ProofWriter depth splits.

This is a lightweight TRUA-side adapter: it keeps LoGiPT as an external
decoder-LM baseline and only adapts ProofWriter examples into text prompts plus
boolean answer parsing. It does not provide gold proof/trace at test time.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def iter_proofwriter(root: Path, depth: int, split: str, limit: int | None = None):
    path = root / "OWA" / f"depth-{depth}" / f"meta-{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            theory = str(item.get("theory", "")).strip()
            for qid, q in item.get("questions", {}).items():
                if "answer" not in q:
                    continue
                yield {
                    "id": f"{item.get('id', path.stem)}::{qid}",
                    "theory": theory,
                    "question": str(q.get("question", "")).strip(),
                    "answer": bool(q["answer"]),
                    "qdep": int(q.get("QDep", item.get("maxD", depth))),
                    "max_depth": int(item.get("maxD", depth)),
                    "source": str(path),
                }
                seen += 1
                if limit is not None and seen >= limit:
                    return


def make_prompt(example: dict[str, Any], tokenizer) -> str:
    user = (
        "You are a deductive reasoning model. Given the facts and rules, answer "
        "the question with exactly one word: True or False.\n\n"
        f"Facts and rules:\n{example['theory']}\n\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<s>[INST] {user} [/INST]"


TRUE_RE = re.compile(r"\b(true|yes|entailed|correct)\b", re.I)
FALSE_RE = re.compile(r"\b(false|no|not entailed|incorrect)\b", re.I)


def parse_bool(text: str) -> bool | None:
    head = text.strip().splitlines()[0] if text.strip() else ""
    false_match = FALSE_RE.search(head) or FALSE_RE.search(text[:256])
    true_match = TRUE_RE.search(head) or TRUE_RE.search(text[:256])
    if false_match and true_match:
        return false_match.start() > true_match.start()
    if false_match:
        return False
    if true_match:
        return True
    return None


def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def evaluate_split(model, tokenizer, examples: list[dict[str, Any]], args) -> dict[str, Any]:
    by_depth: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    preds: list[dict[str, Any]] = []
    correct = 0
    parsed = 0

    for start in tqdm(range(0, len(examples), args.batch_size), desc="LoGiPT eval"):
        batch = examples[start : start + args.batch_size]
        prompts = [make_prompt(x, tokenizer) for x in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_tokens)
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
        for ex, text in zip(batch, texts):
            pred = parse_bool(text)
            ok = pred is not None and pred == ex["answer"]
            parsed += int(pred is not None)
            correct += int(ok)
            by_depth[ex["qdep"]][0] += int(ok)
            by_depth[ex["qdep"]][1] += 1
            if args.keep_predictions:
                preds.append(
                    {
                        "id": ex["id"],
                        "qdep": ex["qdep"],
                        "gold": ex["answer"],
                        "pred": pred,
                        "generation": text,
                    }
                )

    total = len(examples)
    result = {
        "accuracy": correct / max(total, 1),
        "parsed_ratio": parsed / max(total, 1),
        "total": total,
        "by_qdep": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_depth.items())},
        "by_qdep_count": {str(k): v[1] for k, v in sorted(by_depth.items())},
    }
    if args.keep_predictions:
        result["predictions"] = preds
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="proofwriter")
    parser.add_argument("--root", default="data/proofwriter/raw/proofwriter-dataset-V2020.12.3")
    parser.add_argument("--split", default="depth3,depth5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit-per-depth", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--keep-predictions", action="store_true")
    args = parser.parse_args()

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

    root = Path(args.root)
    results: dict[str, Any] = {
        "method": "LoGiPT",
        "model": args.model,
        "dataset": "ProofWriter",
        "setting": "raw theory + question prompt; no gold proof/trace at test time",
        "splits": {},
    }
    for token in [x.strip() for x in args.split.split(",") if x.strip()]:
        depth = int(token.lower().replace("depth", "").replace("-", ""))
        examples = list(iter_proofwriter(root, depth, "test", args.limit_per_depth or None))
        results["splits"][f"depth-{depth}"] = evaluate_split(model, tokenizer, examples, args)

    totals = []
    for split_result in results["splits"].values():
        totals.extend([split_result["accuracy"]] * split_result["total"])
    results["macro_depth_accuracy"] = mean([x["accuracy"] for x in results["splits"].values()])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "splits"}, indent=2))
    for name, split_result in results["splits"].items():
        print(name, {k: v for k, v in split_result.items() if k != "predictions"})


if __name__ == "__main__":
    main()
