#!/usr/bin/env python3
"""NoRA-1.1 text/trace pilot runner for TRUA.

This is intentionally standalone. It converts the official NoRA-1.1 parquet
files into symbolic text examples and compares the same encoder architecture
with and without trace supervision over story facts.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def parse_literal(value, default):
    if isinstance(value, (list, tuple, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return default


ATOM_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)")


def atom_strings(text: str) -> set[str]:
    atoms = set()
    for rel, args in ATOM_RE.findall(str(text)):
        args = ",".join(part.strip() for part in args.split(","))
        atoms.add(f"{rel}({args})")
    return atoms


def relation_words(rel: str) -> str:
    return rel.replace("_", " ")


def edge_to_sentence(edge, rel: str) -> str:
    src, dst = edge
    if src == dst and rel.startswith("is_"):
        return f"entity {src} has property {relation_words(rel)}."
    if src == dst and rel.startswith("no_"):
        return f"entity {src} has property {relation_words(rel)}."
    return f"entity {src} is {relation_words(rel)} entity {dst}."


def load_split(path: Path, label_vocab: dict[str, int] | None = None, limit: int | None = None):
    df = pd.read_parquet(path)
    if limit:
        df = df.iloc[:limit].copy()
    rows = []
    labels_seen = set()
    for _, row in df.iterrows():
        edges = parse_literal(row["story_edges"], [])
        edge_types = parse_literal(row["edge_types"], [])
        query_edge = parse_literal(row["query_edge"], (0, 0))
        query_labels = parse_literal(row["query_label"], [])
        if isinstance(query_labels, str):
            query_labels = [query_labels]
        labels_seen.update(query_labels)
        sentences = [edge_to_sentence(edge, rel) for edge, rel in zip(edges, edge_types)]
        facts = {f"{rel}({edge[0]},{edge[1]})": i for i, (edge, rel) in enumerate(zip(edges, edge_types))}
        trace_atoms = atom_strings(row.get("derivation_chain", ""))
        trace = [0.0] * len(sentences)
        for atom in trace_atoms:
            idx = facts.get(atom)
            if idx is not None:
                trace[idx] = 1.0
        qsrc, qdst = query_edge
        query = f"Which relations hold from entity {qsrc} to entity {qdst}?"
        rows.append(
            {
                "sentences": sentences,
                "context": " ".join(sentences),
                "query": query,
                "labels": list(query_labels),
                "trace": trace,
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
    def __init__(self, samples, tokenizer, max_sents: int, max_text_len: int, max_sent_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_sents = max_sents
        self.max_text_len = max_text_len
        self.max_sent_len = max_sent_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        sentences = x["sentences"][: self.max_sents]
        trace = x["trace"][: self.max_sents]
        if not sentences:
            sentences = ["empty story."]
            trace = [0.0]
        return {**x, "sentences": sentences, "trace": trace}

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
        for i, x in enumerate(batch):
            n = sent_lens[i]
            sent_mask[i, :n] = True
            trace[i, :n] = torch.tensor(x["trace"], dtype=torch.float32)
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
            "target": torch.stack([x["target"] for x in batch]),
            "depth": torch.tensor([x["depth"] for x in batch], dtype=torch.float32),
            "opec": torch.tensor([x["opec"] for x in batch], dtype=torch.float32),
            "bl": torch.tensor([x["bl"] for x in batch], dtype=torch.float32),
        }


class NoraTrua(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_encoder: bool = False, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.sent_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
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
        scores = (sent_vec * query_vec.unsqueeze(1)).sum(-1) / math.sqrt(hidden)
        scores = scores.masked_fill(~batch["sent_mask"], -1e4)
        attn = torch.softmax(scores, dim=-1)
        evidence = (attn.unsqueeze(-1) * sent_vec).sum(1)
        logits = self.classifier(torch.cat([text_vec, evidence], dim=-1))
        return logits, scores


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
        text_vec = self.encoder(
            input_ids=batch["text_ids"],
            attention_mask=batch["text_mask"],
        ).last_hidden_state[:, 0]
        return self.classifier(text_vec), None


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def trace_loss(scores, mask, trace):
    valid = mask & (trace >= 0)
    target = trace * valid.float()
    denom = target.sum(dim=1, keepdim=True).clamp_min(1.0)
    target = target / denom
    logp = torch.log_softmax(scores.masked_fill(~valid, -1e4), dim=-1)
    active = (trace.sum(dim=1) > 0).float()
    loss = -(target * logp).sum(dim=1)
    return (loss * active).sum() / active.sum().clamp_min(1.0)


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


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_gold = [], []
    grouped = {
        "depth": defaultdict(lambda: [[], []]),
        "opec": defaultdict(lambda: [[], []]),
        "bl": defaultdict(lambda: [[], []]),
    }
    trace_hit = trace_total = 0
    for batch in loader:
        batch = move(batch, device)
        logits, scores = model(batch)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        empty = pred.sum(dim=1) == 0
        if empty.any():
            pred[empty, logits[empty].argmax(dim=1)] = 1.0
        gold = batch["target"].float()
        all_pred.append(pred.cpu())
        all_gold.append(gold.cpu())
        if scores is not None:
            top = scores.argmax(dim=1)
            for i, j in enumerate(top.detach().cpu().tolist()):
                if batch["trace"][i].sum().item() > 0:
                    trace_total += 1
                    trace_hit += int(batch["trace"][i, j].item() > 0)
        for i in range(pred.size(0)):
            p = pred[i].detach().cpu()
            y = gold[i].detach().cpu()
            grouped["depth"][bucket_depth(batch["depth"][i].item())][0].append(p)
            grouped["depth"][bucket_depth(batch["depth"][i].item())][1].append(y)
            grouped["opec"][bucket_opec(batch["opec"][i].item())][0].append(p)
            grouped["opec"][bucket_opec(batch["opec"][i].item())][1].append(y)
            grouped["bl"][bucket_bl(batch["bl"][i].item())][0].append(p)
            grouped["bl"][bucket_bl(batch["bl"][i].item())][1].append(y)
    pred = torch.cat(all_pred).numpy().astype(int)
    gold = torch.cat(all_gold).numpy().astype(int)
    exact = float((pred == gold).all(axis=1).mean())
    out = {
        "exact_match": exact,
        "micro_f1": float(f1_score(gold, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "n": int(len(gold)),
        "trace_top1": trace_hit / max(trace_total, 1),
    }
    for group_name, group_values in grouped.items():
        out[f"by_{group_name}"] = {}
        for key, (preds, golds) in sorted(group_values.items(), key=lambda kv: kv[0]):
            p = torch.stack(preds).numpy().astype(int)
            y = torch.stack(golds).numpy().astype(int)
            out[f"by_{group_name}"][key] = {
                "exact_match": float((p == y).all(axis=1).mean()),
                "micro_f1": float(f1_score(y, p, average="micro", zero_division=0)),
                "n": int(len(y)),
            }
    return out


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.root)
    train, label_vocab = load_split(root / "data/train-00000-of-00001.parquet", limit=args.limit_train)
    tests = {}
    for split in ["test_d_na", "test_bl_na", "test_opec_na"]:
        path = root / f"data/{split}-00000-of-00001.parquet"
        tests[split], _ = load_split(path, label_vocab=label_vocab, limit=args.limit_test)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_ds = NoraDataset(train, tokenizer, args.max_sents, args.max_text_len, args.max_sent_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.architecture == "vanilla":
        model = NoraVanilla(args.model_name, len(label_vocab), freeze_encoder=args.freeze_encoder).to(device)
    else:
        model = NoraTrua(args.model_name, len(label_vocab), freeze_encoder=args.freeze_encoder).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    train_log = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move(batch, device)
            logits, scores = model(batch)
            loss = bce(logits, batch["target"])
            if args.lambda_trace > 0 and scores is not None:
                loss = loss + args.lambda_trace * trace_loss(scores, batch["sent_mask"], batch["trace"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}", flush=True)
        train_log.append({"epoch": epoch + 1, "loss": mean_loss})
    results = {}
    for name, samples in tests.items():
        ds = NoraDataset(samples, tokenizer, args.max_sents, args.max_text_len, args.max_sent_len)
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=ds.collate)
        results[name] = evaluate(model, loader, device)
        print(name, json.dumps(results[name], sort_keys=True), flush=True)
    payload = {
        "dataset": "NoRA-1.1",
        "model_name": args.model_name,
        "architecture": args.architecture,
        "seed": args.seed,
        "epochs": args.epochs,
        "lambda_trace": args.lambda_trace,
        "limit_train": args.limit_train,
        "limit_test": args.limit_test,
        "num_labels": len(label_vocab),
        "label_vocab": label_vocab,
        "train_log": train_log,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-name", default="/vepfs/tsra_models/hf/deberta-v3-base")
    parser.add_argument("--architecture", choices=["vanilla", "trua", "tsra"], default="trua")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-trace", type=float, default=0.0)
    parser.add_argument("--max-sents", type=int, default=80)
    parser.add_argument("--max-text-len", type=int, default=384)
    parser.add_argument("--max-sent-len", type=int, default=48)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
