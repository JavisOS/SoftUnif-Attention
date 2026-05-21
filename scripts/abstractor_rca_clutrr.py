#!/usr/bin/env python3
"""PyTorch RCA/Abstractor-style baseline for CLUTRR.

The ICLR 2024 Abstractor codebase is TensorFlow/Keras and targets synthetic
relational tasks. This runner ports the key inductive bias, relational
cross-attention over sentence symbols, to CLUTRR, the closest dataset in this
repo to explicit relational reasoning.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys_path_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(sys_path_root))
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as rel_map


def split_story(text: str):
    return [s.strip() for s in str(text).split(".") if s.strip()]


def query_text(q):
    try:
        a, b = ast.literal_eval(q)
        return f"What is the relation between {a} and {b}?"
    except Exception:
        return str(q)


class ClutrrTextDataset(Dataset):
    def __init__(self, files, limit=None):
        rows = []
        for f in files:
            df = pd.read_csv(f)
            rows.extend(df.to_dict("records"))
        if limit:
            rows = rows[:limit]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            hops = len(ast.literal_eval(r["story_edges"]))
        except Exception:
            hops = -1
        return {
            "sentences": split_story(r["story"]),
            "query": query_text(r["query"]),
            "label": rel_map.get(r["target"], rel_map["nothing"]),
            "hops": hops,
        }


class Collator:
    def __init__(self, tokenizer, max_sents=8, max_len=80):
        self.tokenizer = tokenizer
        self.max_sents = max_sents
        self.max_len = max_len

    def __call__(self, items):
        sent_lens = []
        flat = []
        for x in items:
            sents = (x["sentences"] or [""])[: self.max_sents]
            sent_lens.append(len(sents))
            flat.extend(sents)
        sent_tok = self.tokenizer(flat, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        query_tok = self.tokenizer([x["query"] for x in items], padding=True, truncation=True, max_length=48, return_tensors="pt")
        max_s = max(sent_lens)
        mask = torch.zeros(len(items), max_s, dtype=torch.bool)
        for i, n in enumerate(sent_lens):
            mask[i, :n] = True
        return {
            "sent_ids": sent_tok["input_ids"],
            "sent_mask": sent_tok["attention_mask"],
            "query_ids": query_tok["input_ids"],
            "query_mask": query_tok["attention_mask"],
            "sent_lens": torch.tensor(sent_lens),
            "mask": mask,
            "labels": torch.tensor([x["label"] for x in items], dtype=torch.long),
            "hops": torch.tensor([x["hops"] for x in items], dtype=torch.long),
        }


class RelationalCrossAttention(nn.Module):
    def __init__(self, hidden, rel_dim=128):
        super().__init__()
        self.rel = nn.Sequential(nn.Linear(hidden * 2, rel_dim), nn.ReLU(), nn.Linear(rel_dim, hidden))
        self.q = nn.Linear(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, symbols, query, mask):
        b, n, h = symbols.shape
        left = symbols.unsqueeze(2).expand(b, n, n, h)
        right = symbols.unsqueeze(1).expand(b, n, n, h)
        relations = self.rel(torch.cat([left, right], dim=-1))
        rel_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        q = self.q(query).view(b, 1, 1, h)
        scores = (relations * q).sum(-1) / (h ** 0.5)
        scores = scores.masked_fill(~rel_mask, -1e4)
        attn = torch.softmax(scores.view(b, -1), dim=-1).view(b, n, n)
        ctx = (attn.unsqueeze(-1) * relations).sum(dim=(1, 2))
        pooled = (symbols * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).float()
        return self.out(torch.cat([pooled, ctx], dim=-1))


class AbstractorClutrr(nn.Module):
    def __init__(self, model_name, num_labels, freeze_encoder=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        h = self.encoder.config.hidden_size
        self.rca = RelationalCrossAttention(h)
        self.cls = nn.Linear(h, num_labels)

    def encode(self, ids, mask):
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

    def forward(self, batch):
        flat = self.encode(batch["sent_ids"], batch["sent_mask"])
        query = self.encode(batch["query_ids"], batch["query_mask"])
        b = batch["labels"].size(0)
        n = batch["mask"].size(1)
        h = flat.size(-1)
        symbols = torch.zeros(b, n, h, device=flat.device)
        cur = 0
        for i, ln in enumerate(batch["sent_lens"].tolist()):
            symbols[i, :ln] = flat[cur : cur + ln]
            cur += ln
        rep = self.rca(symbols, query, batch["mask"])
        return self.cls(rep)


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
            for h, p, y in zip(batch["hops"].cpu().tolist(), pred.cpu().tolist(), batch["labels"].cpu().tolist()):
                by_hop[h][1] += 1
                by_hop[h][0] += int(p == y)
    short = sum(v[0] for k, v in by_hop.items() if k <= 3) / max(sum(v[1] for k, v in by_hop.items() if k <= 3), 1)
    long = sum(v[0] for k, v in by_hop.items() if k >= 6) / max(sum(v[1] for k, v in by_hop.items() if k >= 6), 1)
    return {
        "overall": correct / max(total, 1),
        "short_hop": short,
        "long_hop": long,
        "by_hop": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_hop.items())},
        "total": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/data_089907f8")
    ap.add_argument("--model-name", default="microsoft/deberta-base")
    ap.add_argument("--limit-train", type=int, default=1000)
    ap.add_argument("--limit-test", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--out", default="outputs/external_baselines/abstractor_rca_clutrr.json")
    args = ap.parse_args()

    root = Path(args.root)
    train_files = [root / "1.2,1.3_train.csv"]
    test_files = [root / f"1.{i}_test.csv" for i in range(2, 11)]
    train = ClutrrTextDataset(train_files, args.limit_train)
    test = ClutrrTextDataset(test_files, args.limit_test)
    tok = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    collate = Collator(tok)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AbstractorClutrr(args.model_name, len(rel_map), freeze_encoder=args.freeze_encoder).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"Abstractor/RCA epoch {ep+1}"):
            batch = move(batch, device)
            loss = nn.functional.cross_entropy(model(batch), batch["labels"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"epoch={ep+1} loss={loss_sum / max(len(train_loader), 1):.4f}")
    result = {
        "method": "Abstractor/RCA adapted PyTorch",
        "paper": "Abstractors and Relational Cross-Attention, ICLR 2024",
        "dataset": "CLUTRR data_089907f8",
        "train_examples": len(train),
        "test_examples": len(test),
        "results": evaluate(model, test_loader, device),
        "note": "PyTorch port of relational cross-attention inductive bias; official repo is TensorFlow and does not include CLUTRR.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
