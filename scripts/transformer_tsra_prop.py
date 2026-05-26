#!/usr/bin/env python3
"""Transformer TSRA-Prop runner for ProofWriter/RuleTaker/PrOntoQA.

This upgrades the earlier BOW smoke runner to a shared DeBERTa encoder. It
keeps TSRA's key constraint: gold trace supervises sentence-selection logits
only during training; evaluation uses raw context/query text.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generic_tsra_prop import load_prontoqa, load_proofwriter, load_ruletaker_gfair, load_ruletaker_raw


class TextTraceDataset(Dataset):
    def __init__(self, samples, tokenizer, max_sents=12, max_len=160):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_sents = max_sents
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        sentences = (x["sentences"] or [x["context"][:300]])[: self.max_sents]
        trace = (x["trace_labels"] or [0])[: self.max_sents]
        if len(trace) < len(sentences):
            trace = trace + [0] * (len(sentences) - len(trace))
        return {
            "text": x["context"] + " [SEP] " + x["query"],
            "query": x["query"],
            "sentences": sentences,
            "trace": trace,
            "label": int(x.get("label", 1)),
            "depth": int(x.get("depth", -1)),
        }

    def collate(self, batch):
        tok = self.tokenizer
        texts = tok([x["text"] for x in batch], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        queries = tok([x["query"] for x in batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
        flat_sents = []
        sent_lens = []
        for x in batch:
            sent_lens.append(len(x["sentences"]))
            flat_sents.extend(x["sentences"])
        sents = tok(flat_sents, padding=True, truncation=True, max_length=96, return_tensors="pt")
        max_s = max(sent_lens)
        mask = torch.zeros(len(batch), max_s, dtype=torch.bool)
        trace = torch.zeros(len(batch), max_s, dtype=torch.float32)
        cursor = 0
        for i, x in enumerate(batch):
            n = sent_lens[i]
            mask[i, :n] = True
            trace[i, :n] = torch.tensor(x["trace"][:n], dtype=torch.float32)
            cursor += n
        return {
            "text_ids": texts["input_ids"],
            "text_mask": texts["attention_mask"],
            "query_ids": queries["input_ids"],
            "query_mask": queries["attention_mask"],
            "sent_ids": sents["input_ids"],
            "sent_mask": sents["attention_mask"],
            "sent_lens": torch.tensor(sent_lens, dtype=torch.long),
            "mask": mask,
            "trace": trace,
            "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
            "depth": torch.tensor([x["depth"] for x in batch], dtype=torch.long),
        }


class TransformerTsraProp(nn.Module):
    def __init__(self, model_name: str, freeze_encoder: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.sent_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return out[:, 0]

    def forward(self, batch):
        text = self.encode_cls(batch["text_ids"], batch["text_mask"])
        query = self.query_proj(self.encode_cls(batch["query_ids"], batch["query_mask"]))
        flat_sent = self.sent_proj(self.encode_cls(batch["sent_ids"], batch["sent_mask"]))
        bsz = batch["label"].size(0)
        max_s = batch["mask"].size(1)
        hidden = flat_sent.size(-1)
        sent = torch.zeros(bsz, max_s, hidden, device=flat_sent.device)
        cursor = 0
        for i, n in enumerate(batch["sent_lens"].tolist()):
            sent[i, :n] = flat_sent[cursor : cursor + n]
            cursor += n
        scores = (sent * query.unsqueeze(1)).sum(-1) / math.sqrt(hidden)
        scores = scores.masked_fill(~batch["mask"], -1e4)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * sent).sum(1)
        logits = self.classifier(torch.cat([text, ctx], dim=-1))
        return logits, scores


def move(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def evaluate(model, loader, device):
    model.eval()
    correct = total = trace_hit = trace_total = 0
    by_depth = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            logits, scores = model(batch)
            pred = logits.argmax(-1)
            correct += (pred == batch["label"]).sum().item()
            total += pred.numel()
            top = scores.masked_fill(~batch["mask"], -1e4).argmax(-1)
            for i, j in enumerate(top.cpu().tolist()):
                valid_trace = batch["trace"][i][batch["mask"][i]]
                if valid_trace.sum().item() > 0:
                    trace_total += 1
                    trace_hit += int(batch["trace"][i, j].item() > 0)
            for d, p, y in zip(batch["depth"].cpu().tolist(), pred.cpu().tolist(), batch["label"].cpu().tolist()):
                by_depth[d][1] += 1
                by_depth[d][0] += int(p == y)
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "trace_top1": trace_hit / max(trace_total, 1),
        "trace_total": trace_total,
        "by_depth": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_depth.items())},
    }


def trace_ce_loss(scores, mask, trace):
    has_trace = (trace * mask.float()).sum(dim=1) > 0
    if not has_trace.any():
        return scores.new_tensor(0.0)
    masked_scores = scores[has_trace].masked_fill(~mask[has_trace], -1e4)
    target = trace[has_trace] * mask[has_trace].float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(target * torch.log_softmax(masked_scores, dim=-1)).sum(dim=-1).mean()


def run(train, tests, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TransformerTsraProp(args.model_name, freeze_encoder=args.freeze_encoder).to(device)
    train_ds = TextTraceDataset(train, tokenizer, args.max_sents, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            logits, scores = model(batch)
            loss = nn.functional.cross_entropy(logits, batch["label"])
            if args.lambda_trace > 0:
                loss = loss + args.lambda_trace * trace_ce_loss(scores, batch["mask"], batch["trace"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch={epoch+1} loss={total_loss / max(len(train_loader), 1):.4f}")
    results = {}
    for name, samples in tests.items():
        ds = TextTraceDataset(samples, tokenizer, args.max_sents, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate)
        results[name] = evaluate(model, loader, device)
    return {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "lambda_trace": args.lambda_trace,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_depths": args.train_depths,
        "test_depths": args.test_depths,
        "train_qdeps": args.train_qdeps,
        "test_qdeps": args.test_qdeps,
        "train": len(train),
        "results": results,
    }


def _parse_ints(value):
    if value is None or value == "":
        return None
    return [int(x) for x in str(value).split(",") if x != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["proofwriter", "ruletaker_gfair", "ruletaker_raw", "prontoqa"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-base")
    parser.add_argument("--lambda-trace", type=float, default=1.0)
    parser.add_argument("--train-depths", default="0,1,2")
    parser.add_argument("--test-depths", default="3,5")
    parser.add_argument("--train-qdeps", default="")
    parser.add_argument("--test-qdeps", default="")
    parser.add_argument("--limit-train", type=int, default=1000)
    parser.add_argument("--limit-test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-sents", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.limit_train is not None and args.limit_train <= 0:
        args.limit_train = None
    if args.limit_test is not None and args.limit_test <= 0:
        args.limit_test = None

    root = Path(args.root)
    if args.dataset == "proofwriter":
        train_depths = _parse_ints(args.train_depths) or []
        test_depths = _parse_ints(args.test_depths) or []
        train = load_proofwriter(root, train_depths, "train", args.limit_train)
        tests = {f"depth-{d}": load_proofwriter(root, [d], "test", args.limit_test) for d in test_depths}
    elif args.dataset == "ruletaker_gfair":
        train = load_ruletaker_gfair(root, "train", args.limit_train)
        tests = {
            "dev": load_ruletaker_gfair(root, "dev", args.limit_test),
            "test": load_ruletaker_gfair(root, "test", args.limit_test),
        }
    elif args.dataset == "ruletaker_raw":
        train_depths = _parse_ints(args.train_depths) or []
        test_depths = _parse_ints(args.test_depths) or []
        train_qdeps = _parse_ints(args.train_qdeps)
        test_qdeps = _parse_ints(args.test_qdeps)
        train = load_ruletaker_raw(root, train_depths, "train", args.limit_train, qdeps=train_qdeps)
        tests = {
            "dev": load_ruletaker_raw(root, test_depths, "dev", args.limit_test, qdeps=test_qdeps),
            "test": load_ruletaker_raw(root, test_depths, "test", args.limit_test, qdeps=test_qdeps),
        }
    else:
        train_files = ["1hop_ProofsOnly_random_noadj.json", "2hop_ProofsOnly_random_noadj.json"]
        test_files = ["3hop_ProofsOnly_random_noadj.json", "4hop_ProofsOnly_random_noadj.json", "4hop_OOD_Composed_random_noadj.json"]
        train = load_prontoqa(root, train_files, args.limit_train)
        tests = {"ood": load_prontoqa(root, test_files, args.limit_test)}
    result = run(train, tests, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
