#!/usr/bin/env python3
"""MAC-style compositional attention baseline for CLUTRR.

This is an adapted reproduction for the CLUTRR setting. The original CLUTRR
baseline repository includes a MAC config, but its environment depends on old
packages that are absent on the dev machine. This runner keeps the core MAC
idea: iterative query-conditioned attention reads over a text encoding.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as REL_MAP


TOKEN_RE = re.compile(r"@ent\d+|[a-z]+|[.,?;:]")


def anonymize_story_and_query(story: str, query: str) -> tuple[str, str]:
    names = []
    for name in re.findall(r"\[(.*?)\]", str(story)):
        if name not in names:
            names.append(name)
    mapping = {name: f"@ent{i}" for i, name in enumerate(names)}
    clean = str(story)
    for name, ent in mapping.items():
        clean = clean.replace(f"[{name}]", ent)
    try:
        a, b = ast.literal_eval(query)
        q = f"relation between {mapping.get(a, a)} and {mapping.get(b, b)}"
    except Exception:
        q = str(query)
    return clean.lower(), q.lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def build(self, texts):
        counter = Counter(tok for text in texts for tok in text)
        for tok, count in sorted(counter.items()):
            if count >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, toks):
        return torch.tensor([self.stoi.get(tok, 1) for tok in toks], dtype=torch.long)


class ClutrrMacDataset(Dataset):
    def __init__(self, files, vocab=None, limit=None):
        rows = []
        for file in files:
            rows.extend(pd.read_csv(file).to_dict("records"))
        self.rows = rows[:limit] if limit else rows
        self.examples = []
        for row in self.rows:
            story, query = anonymize_story_and_query(row["story"], row["query"])
            try:
                hops = len(ast.literal_eval(row["story_edges"]))
            except Exception:
                hops = -1
            self.examples.append({
                "story_tokens": tokenize(story),
                "query_tokens": tokenize(query),
                "label": REL_MAP[row["target"]],
                "hops": hops,
            })
        self.vocab = vocab

    def texts_for_vocab(self):
        for ex in self.examples:
            yield ex["story_tokens"]
            yield ex["query_tokens"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "story": self.vocab.encode(ex["story_tokens"]),
            "query": self.vocab.encode(ex["query_tokens"]),
            "label": torch.tensor(ex["label"], dtype=torch.long),
            "hops": torch.tensor(ex["hops"], dtype=torch.long),
        }


def collate(items):
    stories = pad_sequence([x["story"] for x in items], batch_first=True, padding_value=0)
    queries = pad_sequence([x["query"] for x in items], batch_first=True, padding_value=0)
    story_mask = stories.ne(0)
    query_mask = queries.ne(0)
    return {
        "story": stories,
        "query": queries,
        "story_mask": story_mask,
        "query_mask": query_mask,
        "labels": torch.stack([x["label"] for x in items]),
        "hops": torch.stack([x["hops"] for x in items]),
    }


class MacAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, emb_dim=128, hidden=128, steps=4, dropout=0.1):
        super().__init__()
        self.steps = steps
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.story_enc = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.query_enc = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        dim = hidden * 2
        self.control = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(steps)])
        self.read = nn.Linear(dim * 3, dim)
        self.mem = nn.GRUCell(dim, dim)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, num_labels))

    def masked_mean(self, x, mask):
        return (x * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).float()

    def forward(self, batch):
        story_emb = self.emb(batch["story"])
        query_emb = self.emb(batch["query"])
        story_h, _ = self.story_enc(story_emb)
        query_h, _ = self.query_enc(query_emb)
        q = self.masked_mean(query_h, batch["query_mask"])
        memory = torch.zeros_like(q)
        for i in range(self.steps):
            ctrl = torch.tanh(self.control[i](torch.cat([q, memory], dim=-1)))
            ctrl_exp = ctrl.unsqueeze(1).expand_as(story_h)
            read_in = torch.cat([story_h, ctrl_exp, story_h * ctrl_exp], dim=-1)
            scores = self.read(read_in).sum(-1)
            scores = scores.masked_fill(~batch["story_mask"], -1e4)
            attn = torch.softmax(scores, dim=-1)
            read_vec = (attn.unsqueeze(-1) * story_h).sum(1)
            memory = self.mem(read_vec, memory)
        return self.cls(torch.cat([memory, q], dim=-1))


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    by_hop = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            pred = model(batch).argmax(-1)
            labels = batch["labels"]
            correct += pred.eq(labels).sum().item()
            total += labels.numel()
            for hop, p, y in zip(batch["hops"].cpu().tolist(), pred.cpu().tolist(), labels.cpu().tolist()):
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
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default="outputs/external_baselines/mac_attention_clutrr.json")
    args = ap.parse_args()

    root = Path(args.root)
    train = ClutrrMacDataset([root / "1.2,1.3_train.csv"], limit=args.limit_train)
    test = ClutrrMacDataset([root / f"1.{i}_test.csv" for i in range(2, 11)], limit=args.limit_test)
    vocab = Vocab()
    vocab.build(list(train.texts_for_vocab()))
    train.vocab = vocab
    test.vocab = vocab

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacAttentionClassifier(len(vocab.itos), len(REL_MAP), steps=args.steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"MAC epoch {ep+1}"):
            batch = move(batch, device)
            loss = nn.functional.cross_entropy(model(batch), batch["labels"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch={ep+1} loss={loss_sum / max(len(train_loader), 1):.4f} test={evaluate(model, test_loader, device)}")
        else:
            print(f"epoch={ep+1} loss={loss_sum / max(len(train_loader), 1):.4f}")

    result = {
        "method": "MAC-style compositional attention adapted",
        "paper": "Compositional Attention Networks for Machine Reasoning / CLUTRR official MAC baseline family",
        "dataset": "CLUTRR data_089907f8",
        "train_examples": len(train),
        "test_examples": len(test),
        "vocab_size": len(vocab.itos),
        "results": evaluate(model, test_loader, device),
        "note": "Local lightweight MAC-style implementation because the official CLUTRR baseline repo requires old unavailable dependencies.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
