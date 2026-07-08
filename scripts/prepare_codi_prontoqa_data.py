#!/usr/bin/env python3
"""Prepare PrOntoQA-OOD JSON files for CODI's official prontoqa branch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def infer_depth(item: dict[str, Any]) -> str:
    source = str(item.get("source_file", ""))
    match = re.search(r"(\d+)hop", source)
    return match.group(1) if match else "unknown"


def convert_item(item: dict[str, Any]) -> dict[str, Any]:
    chain = item.get("trua_chain_of_thought") or item.get("tsra_chain_of_thought") or item.get("steps") or []
    chain = [str(x).strip() for x in chain if str(x).strip()]
    answer = str(item.get("answer", "")).strip()
    if answer and (not chain or chain[-1].rstrip(".") != answer.rstrip(".")):
        chain.append(answer)
    return {
        "question": str(item.get("question", "")).strip(),
        "answer": answer,
        "steps": chain,
        "depth": infer_depth(item),
        "source_file": item.get("source_file", ""),
    }


def convert_file(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    converted = [convert_item(item) for item in data if item.get("question") and item.get("answer")]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    by_depth: dict[str, int] = {}
    for item in converted:
        by_depth[item["depth"]] = by_depth.get(item["depth"], 0) + 1
    print(json.dumps({"src": str(src), "dst": str(dst), "n": len(converted), "by_depth": by_depth}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/vepfs/tsra_outputs/recent_external_data/prontoqa_ood")
    parser.add_argument("--output-dir", default="/vepfs/tsra_outputs/recent_external_data/codi_prontoqa")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    mapping = {
        "train": "prontoqa_ood_coconut_train.json",
        "valid": "prontoqa_ood_coconut_valid.json",
        "test": "prontoqa_ood_coconut_test.json",
    }
    for split, filename in mapping.items():
        convert_file(in_dir / filename, out_dir / f"prontoqa_{split}.json")


if __name__ == "__main__":
    main()
