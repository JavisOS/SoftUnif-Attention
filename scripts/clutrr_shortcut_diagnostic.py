#!/usr/bin/env python3
"""Pilot shortcut-reasoning diagnostic for CLUTRR.

This is a lightweight CLUTRR adaptation of the EMNLP 2023 shortcut-reasoning
diagnostic idea. It uses token occlusion instead of full IG-based input
reduction, then compares extracted token-label patterns on short-hop IID
examples and long-hop OOD examples.

The script is intended for exploratory analysis, not as an official
reproduction of Haraguchi et al. (2023).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from clutrr.cli.baseline import BaselineCollator, BaselineModel, CLUTRRBaselineDataset
from clutrr.config.relation_schema import ID_RELATION_MAP_21_WITH_NOTHING, RELATION_ID_MAP_21_WITH_NOTHING
from clutrr.data.trua_collator import TruaBatchCollator
from clutrr.data.trua_dataset import TruaClutrrDataset
from clutrr.models.backbones import build_tokenizer
from clutrr.models.trua_model import TruaReasonerModel
from clutrr.utils.parsing import parse_pair_literal
from clutrr.utils.seed import set_seed
from scripts.crest_clutrr_baseline import CrestCLUTRRDataset, CrestCollator


def move_batch(batch, device, skip=("raw_batch",)):
    out = {}
    for k, v in batch.items():
        if k in skip:
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def norm_token(token: str) -> str:
    token = token.replace("Ġ", "").replace("▁", "").replace("##", "")
    token = token.strip().lower()
    token = token.strip("[](){}\"'`.,;:!?")
    token = re.sub(r"\s+", "", token)
    return token


def valid_pattern_token(token: str) -> bool:
    if len(token) < 2:
        return False
    if token in {"cls", "sep", "pad", "mask", "unk", "and", "the", "his", "her", "she", "him", "was", "is", "to"}:
        return False
    return bool(re.search(r"[a-z0-9]", token))


def read_clutrr_raw_items(root: str, dataset: str, split: str):
    dataset_dir = Path(root) / dataset
    items = []
    for path in sorted(dataset_dir.glob(f"*_{split}.csv")):
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                query = parse_pair_literal(row[3])
                if query is None:
                    continue
                try:
                    hops = int(row[10].split(".")[-1])
                except Exception:
                    hops = -1
                label = RELATION_ID_MAP_21_WITH_NOTHING.get(row[5], RELATION_ID_MAP_21_WITH_NOTHING["nothing"])
                items.append({"story": row[2], "query": query, "label": label, "hops": hops})
    return items


def format_query(query):
    return f"{query[0]} and {query[1]}"


def token_set_from_ids(tokenizer, ids):
    toks = tokenizer.convert_ids_to_tokens([int(x) for x in ids])
    return {t for t in (norm_token(tok) for tok in toks) if valid_pattern_token(t)}


def metric_counts(preds, labels):
    total = len(labels)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return {"acc": correct / total if total else 0.0, "correct": correct, "total": total}


class BaselineDiagnosticWrapper:
    def __init__(self, model, tokenizer, model_type, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = device
        self.mask_id = tokenizer.mask_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    def encode_items(self, items):
        return self.tokenizer(
            [x["story"] for x in items],
            [format_query(x["query"]) for x in items],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    def predict_items(self, items, batch_size=64):
        self.model.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                batch_items = items[start : start + batch_size]
                enc = self.encode_items(batch_items).to(self.device)
                logits = self.model(enc.input_ids, enc.attention_mask)["logits"]
                preds = logits.argmax(dim=-1).cpu().tolist()
                probs = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
                for item, ids, pred, prob in zip(batch_items, enc.input_ids.cpu().tolist(), preds, probs):
                    rows.append(
                        {
                            "item": item,
                            "pred": pred,
                            "gold": item["label"],
                            "prob": prob,
                            "tokens": token_set_from_ids(self.tokenizer, ids),
                        }
                    )
        return rows

    def extract_one(self, item, top_k=1, occlusion_batch_size=64, max_positions=160):
        enc = self.encode_items([item]).to(self.device)
        with torch.no_grad():
            logits = self.model(enc.input_ids, enc.attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(probs.argmax().item())
            pred_prob = float(probs[pred].item())

        ids = enc.input_ids[0]
        attn = enc.attention_mask[0]
        candidates = []
        for pos, tok_id in enumerate(ids.tolist()):
            if int(attn[pos].item()) != 1:
                continue
            if tok_id in self.special_ids:
                continue
            tok = norm_token(self.tokenizer.convert_ids_to_tokens([tok_id])[0])
            if valid_pattern_token(tok):
                candidates.append((pos, tok))
            if len(candidates) >= max_positions:
                break
        if not candidates:
            return None

        drops = []
        for start in range(0, len(candidates), occlusion_batch_size):
            chunk = candidates[start : start + occlusion_batch_size]
            batch_ids = ids.repeat(len(chunk), 1)
            batch_attn = attn.repeat(len(chunk), 1)
            for row_idx, (pos, _) in enumerate(chunk):
                batch_ids[row_idx, pos] = self.mask_id
            with torch.no_grad():
                logits = self.model(batch_ids.to(self.device), batch_attn.to(self.device))["logits"]
                masked_probs = torch.softmax(logits, dim=-1)[:, pred]
            for (pos, tok), masked_prob in zip(chunk, masked_probs.cpu().tolist()):
                drops.append((pred_prob - float(masked_prob), pos, tok))

        drops.sort(reverse=True)
        selected = []
        seen = set()
        for drop, _, tok in drops:
            if tok in seen:
                continue
            selected.append({"token": tok, "drop": drop})
            seen.add(tok)
            if len(selected) >= top_k:
                break
        if not selected:
            return None
        return {"pattern": tuple(x["token"] for x in selected), "label": pred, "prob": pred_prob, "drops": selected}


class TruaDiagnosticWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.collator = TruaBatchCollator(
            tokenizer,
            model_type=model.model_type,
            unit_encoding_mode=getattr(model, "unit_encoding_mode", "joint"),
        )
        self.mask_id = tokenizer.mask_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    def logits_from_batch(self, batch):
        logits, _, _ = self.model.compute_logits(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            entity_spans=batch["entity_spans"],
            query_indices=batch["query_indices"],
            query_input_ids=batch.get("query_input_ids"),
            query_attention_mask=batch.get("query_attention_mask"),
            path_node_ids=batch.get("path_node_ids"),
        )
        return logits

    def predict_items(self, items, batch_size=32):
        self.model.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, len(items), batch_size):
                batch_items = items[start : start + batch_size]
                batch = move_batch(self.collator(batch_items), self.device)
                logits = self.logits_from_batch(batch)
                preds = logits.argmax(dim=-1).cpu().tolist()
                probs = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
                for item, ids, pred, prob in zip(batch_items, batch["input_ids"].cpu().tolist(), preds, probs):
                    rows.append(
                        {
                            "item": item,
                            "pred": pred,
                            "gold": item["target_id"],
                            "prob": prob,
                            "tokens": token_set_from_ids(self.tokenizer, ids),
                        }
                    )
        return rows

    def extract_one(self, item, top_k=1, occlusion_batch_size=48, max_positions=160):
        self.model.eval()
        batch = move_batch(self.collator([item]), self.device)
        with torch.no_grad():
            logits = self.logits_from_batch(batch)
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(probs.argmax().item())
            pred_prob = float(probs[pred].item())

        ids = batch["input_ids"][0]
        attn = batch["attention_mask"][0]
        candidates = []
        for pos, tok_id in enumerate(ids.tolist()):
            if int(attn[pos].item()) != 1:
                continue
            if tok_id in self.special_ids:
                continue
            tok = norm_token(self.tokenizer.convert_ids_to_tokens([tok_id])[0])
            if valid_pattern_token(tok):
                candidates.append((pos, tok))
            if len(candidates) >= max_positions:
                break
        if not candidates:
            return None

        drops = []
        for start in range(0, len(candidates), occlusion_batch_size):
            chunk = candidates[start : start + occlusion_batch_size]
            rep = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    rep[key] = value.repeat(len(chunk), *([1] * (value.dim() - 1)))
                else:
                    rep[key] = value
            for row_idx, (pos, _) in enumerate(chunk):
                rep["input_ids"][row_idx, pos] = self.mask_id
            with torch.no_grad():
                logits = self.logits_from_batch(rep)
                masked_probs = torch.softmax(logits, dim=-1)[:, pred]
            for (pos, tok), masked_prob in zip(chunk, masked_probs.cpu().tolist()):
                drops.append((pred_prob - float(masked_prob), pos, tok))

        drops.sort(reverse=True)
        selected = []
        seen = set()
        for drop, _, tok in drops:
            if tok in seen:
                continue
            selected.append({"token": tok, "drop": drop})
            seen.add(tok)
            if len(selected) >= top_k:
                break
        if not selected:
            return None
        return {"pattern": tuple(x["token"] for x in selected), "label": pred, "prob": pred_prob, "drops": selected}


def train_vanilla(args, tokenizer, device):
    collator = BaselineCollator(tokenizer, device, model_type=args.model_type)
    train_ds = CLUTRRBaselineDataset(args.root, args.dataset, "train")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    model = BaselineModel(args.model_type, len(RELATION_ID_MAP_21_WITH_NOTHING), model_name_or_path=args.model_name_or_path).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            out = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            out["loss"].backward()
            opt.step()
            total += float(out["loss"].item())
        history.append({"epoch": epoch, "loss": total / max(1, len(loader))})
        print(json.dumps({"variant": "vanilla", **history[-1]}), flush=True)
    return model, history


def train_crest(args, tokenizer, device):
    collator = CrestCollator(tokenizer, device=device, model_type=args.model_type)
    train_ds = CrestCLUTRRDataset(args.root, args.dataset, "train", seed=args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    model = BaselineModel(args.model_type, len(RELATION_ID_MAP_21_WITH_NOTHING), model_name_or_path=args.model_name_or_path).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            base = model(batch["base_input_ids"], batch["base_attention_mask"])
            renamed = model(batch["renamed_input_ids"], batch["renamed_attention_mask"])
            reverse = model(batch["reverse_input_ids"], batch["reverse_attention_mask"])
            main = F.cross_entropy(base["logits"], batch["labels"])
            rename = F.cross_entropy(renamed["logits"], batch["labels"])
            if batch["reverse_mask"].any():
                rev = F.cross_entropy(reverse["logits"][batch["reverse_mask"]], batch["reverse_labels"][batch["reverse_mask"]])
            else:
                rev = torch.zeros((), device=device)
            kl = F.kl_div(
                F.log_softmax(renamed["logits"], dim=-1),
                F.softmax(base["logits"].detach(), dim=-1),
                reduction="batchmean",
            )
            loss = main + args.cf_weight * 0.5 * (rename + rev) + args.consistency_weight * kl
            loss.backward()
            opt.step()
            total += float(loss.item())
        history.append({"epoch": epoch, "loss": total / max(1, len(loader))})
        print(json.dumps({"variant": "crest_style", **history[-1]}), flush=True)
    return model, history


def train_trua(args, tokenizer, device):
    train_ds = TruaClutrrDataset(args.root, args.dataset, "train", tokenizer=tokenizer, augment=True)
    train_ds.data = [x for x in train_ds.data if x is not None]
    collator = TruaBatchCollator(tokenizer, model_type=args.model_type)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    model = TruaReasonerModel(
        device,
        tokenizer,
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        sparse_top_k=0,
        force_gold_edges=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            batch = move_batch(batch, device)
            batch.pop("raw_batch", None)
            opt.zero_grad()
            out = model(batch, lambda1=args.lambda_nexthop, lambda_edge=args.lambda_edge, lambda_cons=args.lambda_consistency)
            out["loss"].backward()
            opt.step()
            total += float(out["loss"].item())
        history.append({"epoch": epoch, "loss": total / max(1, len(loader))})
        print(json.dumps({"variant": "trua", **history[-1]}), flush=True)
    return model, history


def summarize_patterns(patterns, iid_rows, ood_rows, overall_ood_acc, min_iid, min_ood):
    grouped = {}
    for pat in patterns:
        key = (" ".join(pat["pattern"]), int(pat["label"]))
        if key not in grouped or pat["prob"] > grouped[key]["prob"]:
            grouped[key] = dict(pat)

    rows = []
    for (pattern_text, label), pat in grouped.items():
        tokens = set(pattern_text.split())
        iid_support = [r for r in iid_rows if tokens.issubset(r["tokens"])]
        ood_support = [r for r in ood_rows if tokens.issubset(r["tokens"])]
        if len(iid_support) < min_iid or len(ood_support) < min_ood:
            continue
        iid_pred_label = sum(int(r["pred"] == label) for r in iid_support) / len(iid_support)
        iid_acc = sum(int(r["pred"] == label and r["gold"] == label) for r in iid_support) / len(iid_support)
        ood_pred_label = sum(int(r["pred"] == label) for r in ood_support) / len(ood_support)
        ood_acc = sum(int(r["pred"] == r["gold"]) for r in ood_support) / len(ood_support)
        delta = ood_acc - overall_ood_acc
        rows.append(
            {
                "pattern": pattern_text,
                "label": ID_RELATION_MAP_21_WITH_NOTHING.get(label, str(label)),
                "label_id": label,
                "iid_support": len(iid_support),
                "ood_support": len(ood_support),
                "iid_pred_rate": iid_pred_label,
                "iid_acc": iid_acc,
                "ood_pred_rate": ood_pred_label,
                "ood_acc": ood_acc,
                "delta_vs_ood": delta,
                "shortcut_flag": iid_acc >= 0.70 and ood_pred_label >= 0.50 and delta <= -0.05,
                "mean_drop": sum(x["drop"] for x in pat.get("drops", [])) / max(1, len(pat.get("drops", []))),
            }
        )
    rows.sort(key=lambda x: (not x["shortcut_flag"], x["delta_vs_ood"], -x["iid_acc"], -x["ood_pred_rate"]))
    return rows


def run_diagnostic(args, model, tokenizer, device):
    if args.variant in {"trua", "tsra"}:
        test_ds = TruaClutrrDataset(args.root, args.dataset, "test", tokenizer=tokenizer, augment=False)
        test_items = [x for x in test_ds.data if x is not None]
        iid_items = [x for x in test_items if x["hops"] in {2, 3}]
        ood_items = [x for x in test_items if x["hops"] >= 6]
        wrapper = TruaDiagnosticWrapper(model, tokenizer, device)
    else:
        test_items = read_clutrr_raw_items(args.root, args.dataset, "test")
        iid_items = [x for x in test_items if x["hops"] in {2, 3}]
        ood_items = [x for x in test_items if x["hops"] >= 6]
        wrapper = BaselineDiagnosticWrapper(model, tokenizer, args.model_type, device)

    iid_rows = wrapper.predict_items(iid_items, batch_size=args.eval_batch_size)
    ood_rows = wrapper.predict_items(ood_items, batch_size=args.eval_batch_size)
    iid_acc = metric_counts([r["pred"] for r in iid_rows], [r["gold"] for r in iid_rows])["acc"]
    ood_acc = metric_counts([r["pred"] for r in ood_rows], [r["gold"] for r in ood_rows])["acc"]

    correct_iid = [r["item"] for r in iid_rows if r["pred"] == r["gold"]]
    rng = random.Random(args.seed)
    rng.shuffle(correct_iid)
    extract_items = correct_iid[: args.extract_n]

    patterns = []
    for idx, item in enumerate(extract_items, start=1):
        pat = wrapper.extract_one(
            item,
            top_k=args.pattern_top_k,
            occlusion_batch_size=args.occlusion_batch_size,
            max_positions=args.max_positions,
        )
        if pat is not None:
            patterns.append(pat)
        if idx % 20 == 0:
            print(json.dumps({"variant": args.variant, "extracted": idx, "patterns": len(patterns)}), flush=True)

    pattern_rows = summarize_patterns(patterns, iid_rows, ood_rows, ood_acc, args.min_iid_support, args.min_ood_support)
    shortcuts = [x for x in pattern_rows if x["shortcut_flag"]]
    return {
        "iid_size": len(iid_rows),
        "ood_size": len(ood_rows),
        "iid_acc": iid_acc,
        "ood_acc": ood_acc,
        "extract_n": len(extract_items),
        "raw_patterns": len(patterns),
        "supported_patterns": len(pattern_rows),
        "shortcut_count": len(shortcuts),
        "shortcut_rate": len(shortcuts) / len(pattern_rows) if pattern_rows else 0.0,
        "avg_shortcut_delta": sum(x["delta_vs_ood"] for x in shortcuts) / len(shortcuts) if shortcuts else 0.0,
        "patterns": pattern_rows[: args.keep_patterns],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["vanilla", "crest", "trua", "tsra"], required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--dataset", default="data_089907f8")
    parser.add_argument("--model_type", default="deberta-v3")
    parser.add_argument("--model_name_or_path", default="/vepfs/tsra_models/hf/deberta-v3-base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cf_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--lambda_nexthop", type=float, default=1.0)
    parser.add_argument("--lambda_edge", type=float, default=1.0)
    parser.add_argument("--lambda_consistency", type=float, default=5.0)
    parser.add_argument("--extract_n", type=int, default=120)
    parser.add_argument("--pattern_top_k", type=int, default=1)
    parser.add_argument("--occlusion_batch_size", type=int, default=64)
    parser.add_argument("--max_positions", type=int, default=160)
    parser.add_argument("--min_iid_support", type=int, default=3)
    parser.add_argument("--min_ood_support", type=int, default=5)
    parser.add_argument("--keep_patterns", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # GPU placement is controlled by the launcher before Python starts.
    # Do not reset CUDA_VISIBLE_DEVICES here: doing so after importing torch can
    # remap all concurrent jobs back to physical GPU0.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(args.model_type, model_name_or_path=args.model_name_or_path)

    if args.variant == "vanilla":
        model, history = train_vanilla(args, tokenizer, device)
    elif args.variant == "crest":
        model, history = train_crest(args, tokenizer, device)
    else:
        model, history = train_trua(args, tokenizer, device)

    diagnostic = run_diagnostic(args, model, tokenizer, device)
    result = {
        "variant": args.variant,
        "method_note": "pilot token-occlusion shortcut diagnostic; not official IG/input-reduction reproduction",
        "paper": "Haraguchi et al. 2023 Findings EMNLP, DOI 10.18653/v1/2023.findings-emnlp.424",
        "dataset": args.dataset,
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "epochs": args.epochs,
        "history": history,
        "diagnostic": diagnostic,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"finished": args.variant, "out": str(out), "diagnostic": {k: v for k, v in diagnostic.items() if k != "patterns"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
