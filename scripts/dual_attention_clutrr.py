#!/usr/bin/env python3
"""Dual Attention Transformer adapted baseline for CLUTRR.

Uses the official `dual_attention` PyTorch package cloned under
external_baselines/dual-attention. CLUTRR is the most suitable dataset in this
repo because it is explicit entity-relation path reasoning.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_baselines" / "dual-attention"))

from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as rel_map
from dual_attention.dual_attention import DualAttention


def split_story(text: str):
    return [s.strip() for s in str(text).split(".") if s.strip()]


def query_text(q):
    try:
        a, b = ast.literal_eval(q)
        return f"What is the relation between {a} and {b}?"
    except Exception:
        return str(q)


class ClutrrDataset(Dataset):
    def __init__(self, files, limit=None):
        rows = []
        for file in files:
            rows.extend(pd.read_csv(file).to_dict("records"))
        self.rows = rows[:limit] if limit else rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            hops = len(ast.literal_eval(row["story_edges"]))
        except Exception:
            hops = -1
        return {
            "sentences": split_story(row["story"]),
            "query": query_text(row["query"]),
            "label": rel_map.get(row["target"], rel_map["nothing"]),
            "hops": hops,
        }


class Collator:
    def __init__(self, tokenizer, max_sents=8):
        self.tokenizer = tokenizer
        self.max_sents = max_sents

    def __call__(self, items):
        flat_sents, lens = [], []
        for item in items:
            sents = (item["sentences"] or [""])[: self.max_sents]
            lens.append(len(sents))
            flat_sents.extend(sents)
        sent_tok = self.tokenizer(flat_sents, padding=True, truncation=True, max_length=80, return_tensors="pt")
        query_tok = self.tokenizer([x["query"] for x in items], padding=True, truncation=True, max_length=48, return_tensors="pt")
        max_len = max(lens)
        mask = torch.zeros(len(items), max_len, dtype=torch.bool)
        for i, n in enumerate(lens):
            mask[i, :n] = True
        return {
            "sent_ids": sent_tok["input_ids"],
            "sent_mask": sent_tok["attention_mask"],
            "query_ids": query_tok["input_ids"],
            "query_mask": query_tok["attention_mask"],
            "lens": torch.tensor(lens),
            "mask": mask,
            "labels": torch.tensor([x["label"] for x in items], dtype=torch.long),
            "hops": torch.tensor([x["hops"] for x in items], dtype=torch.long),
        }


class DualAttentionClutrr(nn.Module):
    def __init__(self, model_name, num_labels, freeze_encoder=True, n_layers=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        h = self.encoder.config.hidden_size
        self.query_proj = nn.Linear(h, h)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "dual": DualAttention(
                    d_model=h,
                    n_heads_sa=4,
                    n_heads_ra=4,
                    dropout=0.1,
                    ra_kwargs={"n_relations": 4},
                    ra_type="relational_attention",
                ),
                "norm1": nn.LayerNorm(h),
                "ff": nn.Sequential(nn.Linear(h, h * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h * 2, h)),
                "norm2": nn.LayerNorm(h),
            })
            for _ in range(n_layers)
        ])
        self.cls = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, num_labels))

    def encode(self, ids, mask):
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

    def forward(self, batch):
        flat = self.encode(batch["sent_ids"], batch["sent_mask"])
        query = self.query_proj(self.encode(batch["query_ids"], batch["query_mask"]))
        bsz = batch["labels"].size(0)
        n = batch["mask"].size(1)
        h = flat.size(-1)
        x = torch.zeros(bsz, n, h, device=flat.device)
        cur = 0
        for i, ln in enumerate(batch["lens"].tolist()):
            x[i, :ln] = flat[cur:cur + ln]
            cur += ln
        symbols = x + query.unsqueeze(1)
        for layer in self.layers:
            out, _, _ = layer["dual"](x, symbols, attn_mask=None)
            x = layer["norm1"](x + out)
            x = layer["norm2"](x + layer["ff"](x))
        pooled = (x * batch["mask"].unsqueeze(-1).float()).sum(1) / batch["mask"].sum(1, keepdim=True).clamp_min(1).float()
        return self.cls(torch.cat([pooled, query], dim=-1))


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    by_hop = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            pred = model(batch).argmax(-1)
            correct += (pred == batch["labels"]).sum().item()
            total += pred.numel()
            for hop, p, y in zip(batch["hops"].cpu().tolist(), pred.cpu().tolist(), batch["labels"].cpu().tolist()):
                by_hop[hop][0] += int(p == y)
                by_hop[hop][1] += 1
    short_n = sum(v[1] for k, v in by_hop.items() if k <= 3)
    long_n = sum(v[1] for k, v in by_hop.items() if k >= 6)
    return {
        "overall": correct / max(total, 1),
        "short_hop": sum(v[0] for k, v in by_hop.items() if k <= 3) / max(short_n, 1),
        "long_hop": sum(v[0] for k, v in by_hop.items() if k >= 6) / max(long_n, 1),
        "by_hop": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_hop.items())},
        "total": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/data_089907f8")
    ap.add_argument("--model-name", default="microsoft/deberta-base")
    ap.add_argument("--limit-train", type=int, default=5000)
    ap.add_argument("--limit-test", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--out", default="outputs/external_baselines/dual_attention_clutrr.json")
    args = ap.parse_args()

    root = Path(args.root)
    train = ClutrrDataset([root / "1.2,1.3_train.csv"], args.limit_train)
    test = ClutrrDataset([root / f"1.{i}_test.csv" for i in range(2, 11)], args.limit_test)
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    collate = Collator(tok)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualAttentionClutrr(args.model_name, len(rel_map), freeze_encoder=args.freeze_encoder).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"DualAttention CLUTRR epoch {ep+1}"):
            batch = move(batch, device)
            loss = nn.functional.cross_entropy(model(batch), batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"epoch={ep+1} loss={loss_sum / max(len(train_loader), 1):.4f}")
    result = {
        "method": "Dual Attention Transformer adapted",
        "paper": "Disentangling and Integrating Relational and Sensory Information in Transformer Architectures",
        "dataset": "CLUTRR data_089907f8",
        "train_examples": len(train),
        "test_examples": len(test),
        "results": evaluate(model, test_loader, device),
        "note": "Uses official dual_attention PyTorch DualAttention module; CLUTRR adapter is local.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
