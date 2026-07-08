#!/usr/bin/env python3
"""Evaluate official CODI weights on the converted TRUA PrOntoQA-OOD view.

CODI's released evaluator ships GSM8K/SVAMP/commonsense loaders. This adapter
keeps the official CODI model path and generation loop, but supplies a
PrOntoQA-OOD JSON view and exact/contains matching for the final conclusion.
No gold proof/trace is provided to the model at test time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
from tqdm.auto import tqdm


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_depth(item: dict[str, Any]) -> str:
    source = str(item.get("source_file", ""))
    match = re.search(r"(\d+)hop", source)
    if match:
        return match.group(1)
    return "unknown"


def build_prompt(question: str, prompt_style: str) -> str:
    if prompt_style == "default":
        return (
            f"{question}\n"
            "Answer the above question. First think step by step and then "
            "answer the final proven statement."
        )
    if prompt_style == "strict_statement":
        return (
            f"{question}\n"
            "Return only the final proven statement in natural language. "
            "Do not answer with a number, option, or explanation."
        )
    if prompt_style == "copy_prove_statement":
        return (
            f"{question}\n"
            "The final answer should be the exact proposition after 'Prove:'. "
            "Return it as a complete sentence, not as a number."
        )
    if prompt_style == "proof_then_statement":
        return (
            f"{question}\n"
            "Write the proof briefly. End with a line formatted exactly as "
            "'The answer is: <final proven statement>'."
        )
    raise ValueError(f"unknown prompt style: {prompt_style}")


def load_examples(path: Path, limit: int | None = None, prompt_style: str = "default") -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for item in raw:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        examples.append(
            {
                "question": build_prompt(question, prompt_style),
                "answer": answer,
                "depth": infer_depth(item),
                "source_file": item.get("source_file", ""),
            }
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def build_codi(args):
    codi_root = Path(args.codi_root).resolve()
    sys.path.insert(0, str(codi_root))
    from src.model import CODI, ModelArguments, TrainingArguments  # type: ignore

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        train=False,
        lora_init=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        ckpt_dir=args.ckpt_dir,
        full_precision=True,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        model_max_length=args.model_max_length,
        bf16=args.bf16,
        use_lora=True,
        num_latent=args.num_latent,
        use_prj=args.use_prj,
        prj_dim=args.prj_dim,
        prj_no_ln=args.prj_no_ln,
        prj_dropout=args.prj_dropout,
        inf_latent_iterations=args.inf_latent_iterations,
        inf_num_iterations=1,
        remove_eos=args.remove_eos,
        greedy=True,
        do_train=False,
        report_to=[],
    )

    task_type = TaskType.CAUSAL_LM
    model_name = args.model_name_or_path.lower()
    if any(name in model_name for name in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in model_name:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in model_name:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"unsupported CODI model family for LoRA target modules: {args.model_name_or_path}")
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    model = CODI(model_args, training_args, lora_config)
    ckpt = Path(args.ckpt_dir)
    if (ckpt / "model.safetensors").exists():
        state_dict = load_file(str(ckpt / "model.safetensors"))
    elif (ckpt / "pytorch_model.bin").exists():
        state_dict = torch.load(str(ckpt / "pytorch_model.bin"), map_location="cpu")
    else:
        raise FileNotFoundError(f"no CODI checkpoint found under {ckpt}")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    return model, training_args


def extract_prediction(text: str, gold: str) -> str:
    text = text.strip()
    for marker in ["The answer is:", "Answer:", "Final answer:", "final proven statement"]:
        if marker.lower() in text.lower():
            idx = text.lower().rfind(marker.lower())
            return text[idx + len(marker) :].strip()
    sentences = [s.strip() for s in re.split(r"[\n]+", text) if s.strip()]
    if sentences:
        return sentences[-1]
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/vepfs/tsra_outputs/recent_external_data/prontoqa_ood/prontoqa_ood_coconut_test.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--codi-root", default="/root/TRUA/external_baselines/CODI")
    parser.add_argument("--model-name-or-path", default="/vepfs/tsra_models/hf/openai-community-gpt2")
    parser.add_argument("--ckpt-dir", default="/vepfs/tsra_models/hf/zen-E-CODI-gpt2")
    parser.add_argument("--output-dir", default="/vepfs/tsra_outputs/recent_external_baselines/codi_tmp")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        choices=["default", "strict_statement", "copy_prove_statement", "proof_then_statement"],
        default="default",
    )
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-latent", type=int, default=6)
    parser.add_argument("--use-prj", action="store_true", default=True)
    parser.add_argument("--prj-dim", type=int, default=768)
    parser.add_argument("--prj-no-ln", action="store_true", default=False)
    parser.add_argument("--prj-dropout", type=float, default=0.0)
    parser.add_argument("--inf-latent-iterations", type=int, default=6)
    parser.add_argument("--remove-eos", action="store_true", default=True)
    parser.add_argument("--keep-predictions", action="store_true")
    args = parser.parse_args()

    examples = load_examples(Path(args.data), args.limit or None, args.prompt_style)
    model, training_args = build_codi(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if isinstance(tokenizer, bool):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            model_max_length=args.model_max_length,
            padding_side="left",
            use_fast=True,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id or tokenizer.convert_tokens_to_ids("[PAD]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.to(torch.bfloat16 if args.bf16 else torch.float16)
    model.eval()

    correct = 0
    by_depth: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    predictions = []

    for start in tqdm(range(0, len(examples), args.batch_size), desc="CODI PrOntoQA eval"):
        batch_examples = examples[start : start + args.batch_size]
        enc = tokenizer(
            [x["question"] for x in batch_examples],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.model_max_length,
        )
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(enc["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(enc["input_ids"].size(0), 2)
        enc["input_ids"] = torch.cat((enc["input_ids"], bot_tensor), dim=1)
        enc["attention_mask"] = torch.cat((enc["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.codi(
                input_ids=enc["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                attention_mask=enc["attention_mask"],
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)
            for _ in range(training_args.inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            if training_args.remove_eos:
                eot_ids = torch.tensor([model.eot_id], dtype=torch.long, device=device)
            else:
                eot_ids = torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)
            output = model.get_embd(model.codi, model.model_name)(eot_ids).unsqueeze(0).expand(enc["input_ids"].size(0), -1, -1)

            finished = torch.zeros(enc["input_ids"].size(0), dtype=torch.bool, device=device)
            pred_tokens = [[] for _ in range(enc["input_ids"].size(0))]
            for _ in range(args.max_new_tokens):
                out = model.codi(
                    inputs_embeds=output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]
                next_token_ids = torch.argmax(logits, dim=-1)
                for b in range(enc["input_ids"].size(0)):
                    if not finished[b]:
                        token_id = next_token_ids[b].item()
                        pred_tokens[b].append(token_id)
                        if token_id == tokenizer.eos_token_id:
                            finished[b] = True
                if finished.all():
                    break
                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1)

        for ex, token_ids in zip(batch_examples, pred_tokens):
            generation = tokenizer.decode(token_ids, skip_special_tokens=True)
            pred = extract_prediction(generation, ex["answer"])
            pred_norm = normalize_text(pred)
            gold_norm = normalize_text(ex["answer"])
            ok = bool(gold_norm) and (pred_norm == gold_norm or gold_norm in normalize_text(generation))
            correct += int(ok)
            by_depth[ex["depth"]][0] += int(ok)
            by_depth[ex["depth"]][1] += 1
            if args.keep_predictions:
                predictions.append(
                    {
                        "depth": ex["depth"],
                        "gold": ex["answer"],
                        "pred": pred,
                        "generation": generation,
                        "correct": ok,
                        "source_file": ex["source_file"],
                    }
                )

    result: dict[str, Any] = {
        "method": "CODI",
        "model_name_or_path": args.model_name_or_path,
        "ckpt_dir": args.ckpt_dir,
        "dataset": "PrOntoQA-OOD",
        "setting": "converted TRUA PrOntoQA-OOD; no gold proof/trace at test time",
        "prompt_style": args.prompt_style,
        "accuracy": correct / max(len(examples), 1),
        "total": len(examples),
        "by_depth": {k: v[0] / max(v[1], 1) for k, v in sorted(by_depth.items())},
        "by_depth_count": {k: v[1] for k, v in sorted(by_depth.items())},
    }
    if args.keep_predictions:
        result["predictions"] = predictions

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
