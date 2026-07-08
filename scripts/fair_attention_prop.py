#!/usr/bin/env python3
"""Fair attention/reasoning architecture baselines for TRUA-Prop datasets.

The runner reuses the TRUA-Prop data loaders for ProofWriter, RuleTaker, and
PrOntoQA-OOD, but trains only with final labels. It is intended for fair
architecture-level comparison: raw text + query at test time, no gold trace,
no gold graph, and no external symbolic solver.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "external_baselines" / "dual-attention"))

from generic_trua_prop import load_prontoqa, load_proofwriter, load_ruletaker_gfair, load_ruletaker_raw

try:
    from dual_attention.dual_attention import DualAttention
except Exception:
    DualAttention = None


class TextTraceDataset(Dataset):
    def __init__(self, samples, tokenizer, max_sents=16, max_len=192):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_sents = max_sents
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sentences = (item["sentences"] or [item["context"][:300]])[: self.max_sents]
        trace = (item["trace_labels"] or [0])[: self.max_sents]
        if len(trace) < len(sentences):
            trace = trace + [0] * (len(sentences) - len(trace))
        return {
            "text": item["context"] + " [SEP] " + item["query"],
            "query": item["query"],
            "sentences": sentences,
            "trace": trace,
            "label": int(item.get("label", 1)),
            "depth": int(item.get("depth", -1)),
        }

    def collate(self, batch):
        tok = self.tokenizer
        texts = tok([x["text"] for x in batch], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        queries = tok([x["query"] for x in batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
        flat_sents, sent_lens = [], []
        for item in batch:
            sent_lens.append(len(item["sentences"]))
            flat_sents.extend(item["sentences"])
        sents = tok(flat_sents, padding=True, truncation=True, max_length=96, return_tensors="pt")
        max_s = max(sent_lens)
        mask = torch.zeros(len(batch), max_s, dtype=torch.bool)
        trace = torch.zeros(len(batch), max_s, dtype=torch.float32)
        for i, item in enumerate(batch):
            n = sent_lens[i]
            mask[i, :n] = True
            trace[i, :n] = torch.tensor(item["trace"][:n], dtype=torch.float32)
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


class BaseSentenceEncoder(nn.Module):
    def __init__(self, model_name: str, freeze_encoder: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return out[:, 0]

    def encode_batch(self, batch):
        text = self.encode_cls(batch["text_ids"], batch["text_mask"])
        query = self.encode_cls(batch["query_ids"], batch["query_mask"])
        flat_sent = self.encode_cls(batch["sent_ids"], batch["sent_mask"])
        bsz, max_s = batch["label"].size(0), batch["mask"].size(1)
        sent = torch.zeros(bsz, max_s, self.hidden, device=flat_sent.device)
        cursor = 0
        for i, n in enumerate(batch["sent_lens"].tolist()):
            sent[i, :n] = flat_sent[cursor : cursor + n]
            cursor += n
        return text, query, sent


class DualAttentionProp(BaseSentenceEncoder):
    def __init__(self, model_name: str, freeze_encoder: bool, num_labels: int = 2, layers: int = 2):
        if DualAttention is None:
            raise RuntimeError("dual_attention package is not importable")
        super().__init__(model_name, freeze_encoder)
        h = self.hidden
        self.query_proj = nn.Linear(h, h)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
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
                    }
                )
                for _ in range(layers)
            ]
        )
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(h * 3, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, num_labels))

    def forward(self, batch):
        text, query_raw, sent = self.encode_batch(batch)
        query = self.query_proj(query_raw)
        x = sent + query.unsqueeze(1)
        for layer in self.layers:
            out, _, _ = layer["dual"](x, sent, attn_mask=None)
            x = layer["norm1"](x + out)
            x = layer["norm2"](x + layer["ff"](x))
        scores = (x * query.unsqueeze(1)).sum(-1) / math.sqrt(x.size(-1))
        scores = scores.masked_fill(~batch["mask"], -1e4)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * x).sum(1)
        logits = self.classifier(torch.cat([text, query, ctx], dim=-1))
        return logits, scores


class RelationalCrossAttention(nn.Module):
    def __init__(self, hidden: int, rel_dim: int = 128):
        super().__init__()
        self.rel = nn.Sequential(nn.Linear(hidden * 2, rel_dim), nn.ReLU(), nn.Linear(rel_dim, hidden))
        self.q = nn.Linear(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, symbols, query, mask):
        bsz, n_sents, hidden = symbols.shape
        left = symbols.unsqueeze(2).expand(bsz, n_sents, n_sents, hidden)
        right = symbols.unsqueeze(1).expand(bsz, n_sents, n_sents, hidden)
        relations = self.rel(torch.cat([left, right], dim=-1))
        rel_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        q = self.q(query).view(bsz, 1, 1, hidden)
        pair_scores = (relations * q).sum(-1) / math.sqrt(hidden)
        pair_scores = pair_scores.masked_fill(~rel_mask, -1e4)
        pair_attn = torch.softmax(pair_scores.view(bsz, -1), dim=-1).view(bsz, n_sents, n_sents)
        ctx = (pair_attn.unsqueeze(-1) * relations).sum(dim=(1, 2))
        pooled = (symbols * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).float()
        sent_scores = pair_scores.max(dim=2).values + pair_scores.max(dim=1).values
        sent_scores = sent_scores.masked_fill(~mask, -1e4)
        return self.out(torch.cat([pooled, ctx], dim=-1)), sent_scores


class AbstractorProp(BaseSentenceEncoder):
    def __init__(self, model_name: str, freeze_encoder: bool, num_labels: int = 2):
        super().__init__(model_name, freeze_encoder)
        h = self.hidden
        self.rca = RelationalCrossAttention(h)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(h * 3, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, num_labels))

    def forward(self, batch):
        text, query, sent = self.encode_batch(batch)
        ctx, scores = self.rca(sent, query, batch["mask"])
        logits = self.classifier(torch.cat([text, query, ctx], dim=-1))
        return logits, scores


class MacProp(BaseSentenceEncoder):
    def __init__(self, model_name: str, freeze_encoder: bool, num_labels: int = 2, steps: int = 4):
        super().__init__(model_name, freeze_encoder)
        h = self.hidden
        self.steps = steps
        self.controls = nn.ModuleList([nn.Linear(h * 2, h) for _ in range(steps)])
        self.read = nn.Linear(h * 3, h)
        self.mem = nn.GRUCell(h, h)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(h * 3, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, num_labels))

    def forward(self, batch):
        text, query, sent = self.encode_batch(batch)
        memory = torch.zeros_like(query)
        scores = None
        for i in range(self.steps):
            control = torch.tanh(self.controls[i](torch.cat([query, memory], dim=-1)))
            control_exp = control.unsqueeze(1).expand_as(sent)
            read_in = torch.cat([sent, control_exp, sent * control_exp], dim=-1)
            scores = self.read(read_in).sum(-1)
            scores = scores.masked_fill(~batch["mask"], -1e4)
            attn = torch.softmax(scores, dim=-1)
            read_vec = (attn.unsqueeze(-1) * sent).sum(1)
            memory = self.mem(read_vec, memory)
        logits = self.classifier(torch.cat([text, query, memory], dim=-1))
        return logits, scores


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


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
            for depth, pred_i, gold_i in zip(batch["depth"].cpu().tolist(), pred.cpu().tolist(), batch["label"].cpu().tolist()):
                by_depth[depth][1] += 1
                by_depth[depth][0] += int(pred_i == gold_i)
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "trace_top1": trace_hit / max(trace_total, 1),
        "trace_total": trace_total,
        "by_depth": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_depth.items())},
    }


def build_model(method, model_name, freeze_encoder):
    if method == "dual_attention":
        return DualAttentionProp(model_name, freeze_encoder)
    if method == "abstractor_rca":
        return AbstractorProp(model_name, freeze_encoder)
    if method == "mac":
        return MacProp(model_name, freeze_encoder)
    raise ValueError(f"unknown method: {method}")


def load_data(args):
    root = Path(args.root)
    if args.dataset == "proofwriter":
        train_depths = _parse_ints(args.train_depths) or []
        test_depths = _parse_ints(args.test_depths) or []
        train = load_proofwriter(root, train_depths, "train", args.limit_train)
        tests = {f"depth-{d}": load_proofwriter(root, [d], "test", args.limit_test) for d in test_depths}
        return train, tests
    if args.dataset == "ruletaker_gfair":
        train = load_ruletaker_gfair(root, "train", args.limit_train)
        tests = {
            "dev": load_ruletaker_gfair(root, "dev", args.limit_test),
            "test": load_ruletaker_gfair(root, "test", args.limit_test),
        }
        return train, tests
    if args.dataset == "ruletaker_raw":
        train_depths = _parse_ints(args.train_depths) or []
        test_depths = _parse_ints(args.test_depths) or []
        train_qdeps = _parse_ints(args.train_qdeps)
        test_qdeps = _parse_ints(args.test_qdeps)
        train = load_ruletaker_raw(root, train_depths, "train", args.limit_train, qdeps=train_qdeps)
        tests = {
            "dev": load_ruletaker_raw(root, test_depths, "dev", args.limit_test, qdeps=test_qdeps),
            "test": load_ruletaker_raw(root, test_depths, "test", args.limit_test, qdeps=test_qdeps),
        }
        return train, tests
    train_files = ["1hop_ProofsOnly_random_noadj.json", "2hop_ProofsOnly_random_noadj.json"]
    test_files = ["3hop_ProofsOnly_random_noadj.json", "4hop_ProofsOnly_random_noadj.json", "4hop_OOD_Composed_random_noadj.json"]
    train = load_prontoqa(root, train_files, args.limit_train)
    tests = {"ood": load_prontoqa(root, test_files, args.limit_test)}
    return train, tests


def run(train, tests, args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(args.method, args.model_name, args.freeze_encoder).to(device)
    train_ds = TextTraceDataset(train, tokenizer, args.max_sents, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            logits, _ = model(batch)
            loss = nn.functional.cross_entropy(logits, batch["label"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        print(f"epoch={epoch + 1} loss={total_loss / max(len(train_loader), 1):.4f}", flush=True)
    results = {}
    for name, samples in tests.items():
        ds = TextTraceDataset(samples, tokenizer, args.max_sents, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate)
        results[name] = evaluate(model, loader, device)
    return {
        "method": args.method,
        "dataset": args.dataset,
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_depths": args.train_depths,
        "test_depths": args.test_depths,
        "train_qdeps": args.train_qdeps,
        "test_qdeps": args.test_qdeps,
        "train": len(train),
        "results": results,
        "fair_setting": "raw text + query; final-label training only; no gold trace/proof/graph/external solver at test time",
    }


def _parse_ints(value):
    if value is None or value == "":
        return None
    return [int(x) for x in str(value).split(",") if x != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dual_attention", "abstractor_rca", "mac"], required=True)
    parser.add_argument("--dataset", choices=["proofwriter", "ruletaker_gfair", "ruletaker_raw", "prontoqa"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-name", default="/vepfs/tsra_models/hf/deberta-base")
    parser.add_argument("--train-depths", default="0,1,2")
    parser.add_argument("--test-depths", default="3,5")
    parser.add_argument("--train-qdeps", default="")
    parser.add_argument("--test-qdeps", default="")
    parser.add_argument("--limit-train", type=int, default=1000)
    parser.add_argument("--limit-test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-sents", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.limit_train is not None and args.limit_train <= 0:
        args.limit_train = None
    if args.limit_test is not None and args.limit_test <= 0:
        args.limit_test = None
    train, tests = load_data(args)
    result = run(train, tests, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
