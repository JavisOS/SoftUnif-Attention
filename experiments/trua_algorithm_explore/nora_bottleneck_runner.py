#!/usr/bin/env python3
"""NoRA-1.1 bottleneck TRUA exploration runner.

This standalone script keeps NoRA experiments isolated from the main TRUA
training code. It compares a vanilla encoder classifier with a multi-step
trace-supervised bottleneck reader under the same threshold calibration.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


ATOM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)")


def parse_literal(value, default):
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return default


def atom_string(rel: str, args: str) -> str:
    args = ",".join(part.strip() for part in args.split(","))
    return f"{rel}({args})"


def first_atom(text: str) -> str | None:
    match = ATOM_RE.search(str(text))
    if not match:
        return None
    return atom_string(match.group(1), match.group(2))


def iter_derivation_texts(value):
    value = parse_literal(value, value)
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_derivation_texts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_derivation_texts(child)
    else:
        yield str(value)


def ordered_fact_atoms(derivation) -> list[str]:
    atoms = []
    for text in iter_derivation_texts(derivation):
        for part in text.split("|"):
            part = part.strip()
            if not part.startswith("fact:"):
                continue
            atom = first_atom(part)
            if atom:
                atoms.append(atom)
    return atoms


def relation_words(rel: str) -> str:
    return rel.replace("_", " ")


def edge_to_sentence(edge, rel: str) -> str:
    src, dst = edge
    if src == dst and (rel.startswith("is_") or rel.startswith("no_")):
        return f"entity {src} has property {relation_words(rel)}."
    return f"entity {src} is {relation_words(rel)} entity {dst}."


def load_split(path: Path, label_vocab: dict[str, int] | None = None, limit: int | None = None):
    df = pd.read_parquet(path)
    if limit:
        df = df.iloc[:limit].copy()

    rows = []
    labels_seen = set()
    pending_labels = []
    for _, row in df.iterrows():
        edges = parse_literal(row["story_edges"], [])
        edge_types = parse_literal(row["edge_types"], [])
        query_edge = parse_literal(row["query_edge"], (0, 0))
        labels = parse_literal(row["query_label"], [])
        if isinstance(labels, str):
            labels = [labels]
        labels = list(labels)
        labels_seen.update(labels)
        pending_labels.append(labels)

        sentences = [edge_to_sentence(edge, rel) for edge, rel in zip(edges, edge_types)]
        fact_to_index = {}
        for i, (edge, rel) in enumerate(zip(edges, edge_types)):
            fact_to_index.setdefault(f"{rel}({edge[0]},{edge[1]})", i)

        trace_steps = []
        used = set()
        for atom in ordered_fact_atoms(row.get("derivation_chain", "")):
            idx = fact_to_index.get(atom)
            if idx is not None and idx not in used:
                trace_steps.append(idx)
                used.add(idx)

        trace = [0.0] * len(sentences)
        for idx in trace_steps:
            trace[idx] = 1.0

        qsrc, qdst = query_edge
        rows.append(
            {
                "sentences": sentences,
                "context": " ".join(sentences),
                "query": f"Which relations hold from entity {qsrc} to entity {qdst}?",
                "labels": labels,
                "trace": trace,
                "trace_steps": trace_steps,
                "depth": float(row["ReasoningDepth"]),
                "opec": float(row["OPEC"]),
                "bl": float(row["BL"]),
            }
        )

    if label_vocab is None:
        label_vocab = {label: i for i, label in enumerate(sorted(labels_seen))}

    for item in rows:
        y = torch.zeros(len(label_vocab), dtype=torch.float32)
        for label in item["labels"]:
            if label in label_vocab:
                y[label_vocab[label]] = 1.0
        item["target"] = y
    return rows, label_vocab


class NoraDataset(Dataset):
    def __init__(
        self,
        samples,
        tokenizer,
        max_sents: int,
        max_text_len: int,
        max_sent_len: int,
        max_trace_steps: int,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_sents = max_sents
        self.max_text_len = max_text_len
        self.max_sent_len = max_sent_len
        self.max_trace_steps = max_trace_steps

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = dict(self.samples[idx])
        sentences = item["sentences"][: self.max_sents]
        trace = item["trace"][: self.max_sents]
        if not sentences:
            sentences = ["empty story."]
            trace = [0.0]
        item["sentences"] = sentences
        item["trace"] = trace
        item["trace_steps"] = [s for s in item["trace_steps"] if s < self.max_sents][: self.max_trace_steps]
        return item

    def collate(self, batch):
        tok = self.tokenizer
        texts = [x["context"] + " [SEP] " + x["query"] for x in batch]
        text_tok = tok(texts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt")
        query_tok = tok([x["query"] for x in batch], padding=True, truncation=True, max_length=64, return_tensors="pt")

        flat_sents, sent_lens = [], []
        for x in batch:
            sent_lens.append(len(x["sentences"]))
            flat_sents.extend(x["sentences"])
        sent_tok = tok(flat_sents, padding=True, truncation=True, max_length=self.max_sent_len, return_tensors="pt")

        max_s = max(sent_lens)
        sent_mask = torch.zeros(len(batch), max_s, dtype=torch.bool)
        trace = torch.zeros(len(batch), max_s, dtype=torch.float32)
        trace_steps = torch.full((len(batch), self.max_trace_steps), -100, dtype=torch.long)
        for i, x in enumerate(batch):
            n = sent_lens[i]
            sent_mask[i, :n] = True
            trace[i, :n] = torch.tensor(x["trace"], dtype=torch.float32)
            for j, step_idx in enumerate(x["trace_steps"][: self.max_trace_steps]):
                trace_steps[i, j] = step_idx

        return {
            "text_ids": text_tok["input_ids"],
            "text_mask": text_tok["attention_mask"],
            "query_ids": query_tok["input_ids"],
            "query_mask": query_tok["attention_mask"],
            "sent_ids": sent_tok["input_ids"],
            "sent_mask_flat": sent_tok["attention_mask"],
            "sent_lens": torch.tensor(sent_lens, dtype=torch.long),
            "sent_mask": sent_mask,
            "trace": trace,
            "trace_steps": trace_steps,
            "target": torch.stack([x["target"] for x in batch]),
            "depth": torch.tensor([x["depth"] for x in batch], dtype=torch.float32),
            "opec": torch.tensor([x["opec"] for x in batch], dtype=torch.float32),
            "bl": torch.tensor([x["bl"] for x in batch], dtype=torch.float32),
        }


class NoraVanilla(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_encoder: bool = False, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, batch):
        text_vec = self.encoder(input_ids=batch["text_ids"], attention_mask=batch["text_mask"]).last_hidden_state[:, 0]
        logits = self.classifier(text_vec)
        return {"fused_logits": logits, "scores": None}


class NoraBottleneckTrua(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_glimpses: int = 4,
        text_residual: float = 0.2,
        bypass_dropout: float = 0.5,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.num_glimpses = num_glimpses
        self.text_residual = text_residual
        self.bypass_dropout = bypass_dropout
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.sent_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.reader = nn.GRUCell(hidden, hidden)
        self.reason_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.text_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def encode_cls(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def forward(self, batch):
        text_vec = self.encode_cls(batch["text_ids"], batch["text_mask"])
        query_vec = self.query_proj(self.encode_cls(batch["query_ids"], batch["query_mask"]))
        flat_sent = self.sent_proj(self.encode_cls(batch["sent_ids"], batch["sent_mask_flat"]))

        bsz, max_s = batch["sent_mask"].shape
        hidden = flat_sent.shape[-1]
        sent_vec = flat_sent.new_zeros((bsz, max_s, hidden))
        cursor = 0
        for i, n in enumerate(batch["sent_lens"].tolist()):
            sent_vec[i, :n] = flat_sent[cursor : cursor + n]
            cursor += n

        state = query_vec
        scores_all, evidence_all = [], []
        for _ in range(self.num_glimpses):
            scores = (sent_vec * state.unsqueeze(1)).sum(-1) / math.sqrt(hidden)
            scores = scores.masked_fill(~batch["sent_mask"], -1e4)
            attn = torch.softmax(scores, dim=-1)
            evidence = (attn.unsqueeze(-1) * sent_vec).sum(1)
            state = self.reader(evidence, state)
            scores_all.append(scores)
            evidence_all.append(evidence)

        scores_stack = torch.stack(scores_all, dim=1)
        evidence_mean = torch.stack(evidence_all, dim=1).mean(1)
        reason_logits = self.reason_classifier(torch.cat([query_vec, state, evidence_mean], dim=-1))
        text_logits = self.text_classifier(text_vec)

        text_scale = self.text_residual
        if self.training and self.bypass_dropout > 0:
            keep = (torch.rand(text_logits.shape[0], 1, device=text_logits.device) > self.bypass_dropout).float()
            text_logits = text_logits * keep
        fused_logits = reason_logits + text_scale * text_logits
        return {
            "fused_logits": fused_logits,
            "reason_logits": reason_logits,
            "text_logits": text_logits,
            "scores": scores_stack,
        }


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def step_trace_loss(scores, sent_mask, trace_steps):
    if scores is None:
        return torch.tensor(0.0, device=sent_mask.device)
    max_steps = min(scores.size(1), trace_steps.size(1))
    losses = []
    for step in range(max_steps):
        target = trace_steps[:, step]
        active = target >= 0
        if active.any():
            losses.append(nn.functional.cross_entropy(scores[active, step, :], target[active]))
    if not losses:
        return torch.tensor(0.0, device=sent_mask.device)
    return torch.stack(losses).mean()


def trace_metrics(scores, trace):
    if scores is None:
        return {"trace_top1_first": 0.0, "trace_any_glimpse": 0.0, "trace_gold_mass": 0.0, "trace_n": 0}
    with torch.no_grad():
        active = trace.sum(dim=1) > 0
        if not active.any():
            return {"trace_top1_first": 0.0, "trace_any_glimpse": 0.0, "trace_gold_mass": 0.0, "trace_n": 0}
        s = scores[active]
        t = trace[active]
        top = s.argmax(dim=-1)
        hits = torch.gather(t.unsqueeze(1).expand(-1, s.size(1), -1), 2, top.unsqueeze(-1)).squeeze(-1) > 0
        attn = torch.softmax(s, dim=-1)
        mass = (attn * t.unsqueeze(1)).sum(-1).mean()
        return {
            "trace_top1_first": float(hits[:, 0].float().mean().item()),
            "trace_any_glimpse": float(hits.any(dim=1).float().mean().item()),
            "trace_gold_mass": float(mass.item()),
            "trace_n": int(active.sum().item()),
        }


def bucket_depth(x):
    return str(int(x))


def bucket_opec(x):
    x = int(x)
    if x <= 0:
        return "0"
    if x <= 2:
        return "1-2"
    if x <= 4:
        return "3-4"
    return "5+"


def bucket_bl(x):
    if x <= 1.0:
        return "<=1.0"
    if x <= 1.5:
        return "1.0-1.5"
    return ">1.5"


def predictions_from_logits(logits, threshold):
    probs = torch.sigmoid(logits)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(probs.device)
    pred = (probs >= threshold).float()
    empty = pred.sum(dim=1) == 0
    if empty.any():
        pred[empty, logits[empty].argmax(dim=1)] = 1.0
    return pred


def score_predictions(gold, pred):
    gold_np = gold.numpy().astype(int)
    pred_np = pred.numpy().astype(int)
    return {
        "exact_match": float((pred_np == gold_np).all(axis=1).mean()),
        "micro_f1": float(f1_score(gold_np, pred_np, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold_np, pred_np, average="macro", zero_division=0)),
    }


@torch.no_grad()
def collect_outputs(model, loader, device):
    model.eval()
    stores = defaultdict(list)
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        for key, value in out.items():
            if torch.is_tensor(value):
                stores[key].append(value.detach().cpu())
        for key in ["target", "depth", "opec", "bl", "trace"]:
            stores[key].append(batch[key].detach().cpu())
    return {key: torch.cat(values, dim=0) for key, values in stores.items()}


def tune_global_threshold(logits, gold, metric: str):
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = predictions_from_logits(logits, float(threshold)).cpu()
        scores = score_predictions(gold, pred)
        value = scores["exact_match"] if metric == "exact_match" else scores["micro_f1"]
        if value > best_score:
            best_threshold, best_score = float(threshold), float(value)
    return best_threshold, best_score


def evaluate_outputs(outputs, threshold):
    gold = outputs["target"].float()
    logits = outputs["fused_logits"]
    pred = predictions_from_logits(logits, threshold).cpu()
    out = score_predictions(gold, pred)
    out["n"] = int(gold.size(0))
    out.update(trace_metrics(outputs.get("scores"), outputs["trace"]))

    for head_key, out_key in [("reason_logits", "reason"), ("text_logits", "text")]:
        if head_key in outputs:
            head_pred = predictions_from_logits(outputs[head_key], threshold).cpu()
            head_scores = score_predictions(gold, head_pred)
            out[f"{out_key}_exact_match"] = head_scores["exact_match"]
            out[f"{out_key}_micro_f1"] = head_scores["micro_f1"]

    grouped = {
        "depth": defaultdict(lambda: [[], []]),
        "opec": defaultdict(lambda: [[], []]),
        "bl": defaultdict(lambda: [[], []]),
    }
    for i in range(pred.size(0)):
        p = pred[i]
        y = gold[i]
        grouped["depth"][bucket_depth(outputs["depth"][i].item())][0].append(p)
        grouped["depth"][bucket_depth(outputs["depth"][i].item())][1].append(y)
        grouped["opec"][bucket_opec(outputs["opec"][i].item())][0].append(p)
        grouped["opec"][bucket_opec(outputs["opec"][i].item())][1].append(y)
        grouped["bl"][bucket_bl(outputs["bl"][i].item())][0].append(p)
        grouped["bl"][bucket_bl(outputs["bl"][i].item())][1].append(y)
    for group_name, group_values in grouped.items():
        out[f"by_{group_name}"] = {}
        for key, (preds, golds) in sorted(group_values.items(), key=lambda kv: kv[0]):
            p = torch.stack(preds)
            y = torch.stack(golds)
            scores = score_predictions(y, p)
            scores["n"] = int(y.size(0))
            out[f"by_{group_name}"][key] = scores
    return out


def make_loaders(samples, tokenizer, args, shuffle: bool):
    ds = NoraDataset(samples, tokenizer, args.max_sents, args.max_text_len, args.max_sent_len, args.max_trace_steps)
    return DataLoader(
        ds,
        batch_size=args.batch_size if shuffle else args.eval_batch_size,
        shuffle=shuffle,
        collate_fn=ds.collate,
        num_workers=args.num_workers,
    )


def compute_pos_weight(samples, num_labels: int, max_weight: float):
    y = torch.stack([s["target"] for s in samples])
    pos = y.sum(dim=0)
    neg = y.size(0) - pos
    weight = neg / pos.clamp_min(1.0)
    return weight.clamp(1.0, max_weight)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.root)
    train, label_vocab = load_split(root / "data/train-00000-of-00001.parquet", limit=args.limit_train)
    tests = {}
    for split in ["test_d_na", "test_bl_na", "test_opec_na"]:
        tests[split], _ = load_split(root / f"data/{split}-00000-of-00001.parquet", label_vocab=label_vocab, limit=args.limit_test)

    rng = random.Random(args.seed)
    indices = list(range(len(train)))
    rng.shuffle(indices)
    calib_n = int(len(indices) * args.calib_ratio)
    calib_indices = indices[:calib_n]
    fit_indices = indices[calib_n:] if calib_n else indices
    fit_samples = [train[i] for i in fit_indices]
    calib_samples = [train[i] for i in calib_indices] if calib_n else fit_samples

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_loader = make_loaders(fit_samples, tokenizer, args, shuffle=True)
    calib_loader = make_loaders(calib_samples, tokenizer, args, shuffle=False)
    test_loaders = {name: make_loaders(samples, tokenizer, args, shuffle=False) for name, samples in tests.items()}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.architecture == "vanilla":
        model = NoraVanilla(args.model_name, len(label_vocab), freeze_encoder=args.freeze_encoder, dropout=args.dropout).to(device)
    else:
        model = NoraBottleneckTrua(
            args.model_name,
            len(label_vocab),
            num_glimpses=args.num_glimpses,
            text_residual=args.text_residual,
            bypass_dropout=args.bypass_dropout,
            freeze_encoder=args.freeze_encoder,
            dropout=args.dropout,
        ).to(device)

    pos_weight = None
    if args.loss == "balanced_bce":
        pos_weight = compute_pos_weight(fit_samples, len(label_vocab), args.max_pos_weight).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    train_log = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move(batch, device)
            out = model(batch)
            loss = bce(out["fused_logits"], batch["target"])
            if "reason_logits" in out and args.lambda_reason > 0:
                loss = loss + args.lambda_reason * bce(out["reason_logits"], batch["target"])
            if "text_logits" in out and args.lambda_text > 0:
                loss = loss + args.lambda_text * bce(out["text_logits"], batch["target"])
            if args.lambda_trace > 0 and out.get("scores") is not None:
                loss = loss + args.lambda_trace * step_trace_loss(out["scores"], batch["sent_mask"], batch["trace_steps"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}", flush=True)
        train_log.append({"epoch": epoch + 1, "loss": mean_loss})

    calib_outputs = collect_outputs(model, calib_loader, device)
    if args.threshold_mode == "global":
        threshold, calib_score = tune_global_threshold(calib_outputs["fused_logits"], calib_outputs["target"], args.calib_metric)
    else:
        threshold, calib_score = args.threshold, None
    calib_result = evaluate_outputs(calib_outputs, threshold)
    print("calib", json.dumps(calib_result, sort_keys=True), flush=True)

    results = {}
    for name, loader in test_loaders.items():
        outputs = collect_outputs(model, loader, device)
        results[name] = evaluate_outputs(outputs, threshold)
        print(name, json.dumps(results[name], sort_keys=True), flush=True)

    card = Counter(int(s["target"].sum().item()) for s in train)
    trace_len = Counter(len(s["trace_steps"]) for s in train)
    payload = {
        "dataset": "NoRA-1.1",
        "model_name": args.model_name,
        "architecture": args.architecture,
        "seed": args.seed,
        "epochs": args.epochs,
        "loss": args.loss,
        "lambda_trace": args.lambda_trace,
        "lambda_reason": args.lambda_reason,
        "lambda_text": args.lambda_text,
        "num_glimpses": args.num_glimpses,
        "text_residual": args.text_residual,
        "bypass_dropout": args.bypass_dropout,
        "limit_train": args.limit_train,
        "limit_test": args.limit_test,
        "calib_ratio": args.calib_ratio,
        "threshold_mode": args.threshold_mode,
        "threshold": threshold,
        "calib_metric": args.calib_metric,
        "calib_score": calib_score,
        "num_labels": len(label_vocab),
        "label_vocab": label_vocab,
        "train_cardinality": dict(card),
        "train_trace_len": dict(trace_len),
        "train_log": train_log,
        "calib_result": calib_result,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/nora_1_1")
    parser.add_argument("--model-name", default="/vepfs/tsra_models/hf/deberta-v3-base")
    parser.add_argument("--architecture", choices=["vanilla", "bottleneck"], default="bottleneck")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["bce", "balanced_bce"], default="balanced_bce")
    parser.add_argument("--max-pos-weight", type=float, default=15.0)
    parser.add_argument("--lambda-trace", type=float, default=0.5)
    parser.add_argument("--lambda-reason", type=float, default=1.0)
    parser.add_argument("--lambda-text", type=float, default=0.1)
    parser.add_argument("--num-glimpses", type=int, default=4)
    parser.add_argument("--text-residual", type=float, default=0.2)
    parser.add_argument("--bypass-dropout", type=float, default=0.5)
    parser.add_argument("--threshold-mode", choices=["fixed", "global"], default="global")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calib-metric", choices=["exact_match", "micro_f1"], default="exact_match")
    parser.add_argument("--calib-ratio", type=float, default=0.15)
    parser.add_argument("--max-sents", type=int, default=96)
    parser.add_argument("--max-text-len", type=int, default=384)
    parser.add_argument("--max-sent-len", type=int, default=48)
    parser.add_argument("--max-trace-steps", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
