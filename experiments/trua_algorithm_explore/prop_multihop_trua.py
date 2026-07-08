#!/usr/bin/env python3
"""Exploratory multi-hop TRUA-Prop diagnostic.

This file is intentionally isolated from the main TRUA training scripts.  It
tests whether a recurrent evidence reader couples trace supervision to the
final prediction more strongly than a single evidence-attention pass.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from clutrr.models.trua_core import masked_trace_distribution_loss
from generic_trua_prop import load_prontoqa, load_proofwriter, load_ruletaker_gfair, load_ruletaker_raw


class TextTraceDataset(Dataset):
    def __init__(self, samples, tokenizer, max_sents=16, max_len=192):
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
        for i, x in enumerate(batch):
            n = sent_lens[i]
            mask[i, :n] = True
            trace[i, :n] = torch.tensor(x["trace"][:n], dtype=torch.float32)

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


class MultiHopTruaProp(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        reasoner: str = "multihop",
        num_glimpses: int = 4,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if reasoner not in {"single", "multihop"}:
            raise ValueError(f"unsupported reasoner: {reasoner}")
        self.reasoner = reasoner
        self.num_glimpses = max(1, num_glimpses if reasoner == "multihop" else 1)
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.sent_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.state_norm = nn.LayerNorm(hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.hop_gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return out[:, 0]

    def forward(self, batch, coverage_penalty: float = 0.0):
        text = self.encode_cls(batch["text_ids"], batch["text_mask"])
        query = self.query_proj(self.encode_cls(batch["query_ids"], batch["query_mask"]))
        flat_sent = self.sent_proj(self.encode_cls(batch["sent_ids"], batch["sent_mask"]))

        bsz = batch["label"].size(0)
        max_s = batch["mask"].size(1)
        hidden = flat_sent.size(-1)
        sent = torch.zeros(bsz, max_s, hidden, device=flat_sent.device, dtype=flat_sent.dtype)
        cursor = 0
        for i, n in enumerate(batch["sent_lens"].tolist()):
            sent[i, :n] = flat_sent[cursor : cursor + n]
            cursor += n

        state = self.state_norm(query)
        coverage = sent.new_zeros(bsz, max_s)
        scores_list = []
        attn_list = []
        ctx_list = []
        for _ in range(self.num_glimpses):
            scores = (sent * state.unsqueeze(1)).sum(-1) / math.sqrt(hidden)
            if coverage_penalty > 0:
                scores = scores - coverage_penalty * coverage
            scores = scores.masked_fill(~batch["mask"], -1e4)
            attn = torch.softmax(scores, dim=-1)
            ctx = (attn.unsqueeze(-1) * sent).sum(1)
            state = self.state_norm(self.gru(ctx, state))
            coverage = coverage + attn
            scores_list.append(scores)
            attn_list.append(attn)
            ctx_list.append(ctx)

        ctx_stack = torch.stack(ctx_list, dim=1)
        hop_weights = torch.softmax(self.hop_gate(torch.cat([ctx_stack, state.unsqueeze(1).expand_as(ctx_stack)], dim=-1)).squeeze(-1), dim=-1)
        ctx_weighted = (hop_weights.unsqueeze(-1) * ctx_stack).sum(1)
        ctx_max = ctx_stack.max(dim=1).values
        logits = self.classifier(torch.cat([text, state, ctx_weighted, ctx_max], dim=-1))
        return logits, scores_list, attn_list


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def trace_losses(scores_list, attn_list, mask, trace, coverage_weight: float, offtrace_weight: float):
    step_losses = [masked_trace_distribution_loss(scores, mask, trace) for scores in scores_list]
    step_loss = torch.stack(step_losses).mean() if step_losses else trace.new_tensor(0.0)

    has_trace = (trace * mask.float()).sum(dim=1) > 0
    if not bool(has_trace.any()):
        return step_loss, trace.new_tensor(0.0), trace.new_tensor(0.0)

    attn_stack = torch.stack(attn_list, dim=1)
    coverage = attn_stack.sum(dim=1) * mask.float()
    coverage_dist = coverage / coverage.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target = trace * mask.float()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    cov_loss = -(target[has_trace] * torch.log(coverage_dist[has_trace].clamp_min(1e-8))).sum(dim=-1).mean()

    nontrace = (mask.float() * (1.0 - trace.float())).clamp(min=0.0)
    offtrace = (attn_stack * nontrace.unsqueeze(1)).sum(dim=-1).mean()
    return step_loss, coverage_weight * cov_loss, offtrace_weight * offtrace


def evaluate(model, loader, device, coverage_penalty: float):
    model.eval()
    correct = total = trace_top1 = trace_any = trace_total = 0
    gold_mass_sum = 0.0
    by_depth = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            logits, scores_list, attn_list = model(batch, coverage_penalty=coverage_penalty)
            pred = logits.argmax(-1)
            correct += (pred == batch["label"]).sum().item()
            total += pred.numel()

            top_each = [scores.masked_fill(~batch["mask"], -1e4).argmax(-1) for scores in scores_list]
            attn_stack = torch.stack(attn_list, dim=1)
            for i in range(pred.size(0)):
                valid_trace = batch["trace"][i][batch["mask"][i]]
                if valid_trace.sum().item() <= 0:
                    continue
                trace_total += 1
                trace_top1 += int(batch["trace"][i, top_each[0][i]].item() > 0)
                trace_any += int(any(batch["trace"][i, top[i]].item() > 0 for top in top_each))
                gold_mass_sum += float((attn_stack[i] * batch["trace"][i].unsqueeze(0)).sum(dim=-1).mean().item())

            for d, p, y in zip(batch["depth"].cpu().tolist(), pred.cpu().tolist(), batch["label"].cpu().tolist()):
                by_depth[d][1] += 1
                by_depth[d][0] += int(p == y)
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "trace_top1_first": trace_top1 / max(trace_total, 1),
        "trace_any_glimpse": trace_any / max(trace_total, 1),
        "trace_gold_mass": gold_mass_sum / max(trace_total, 1),
        "trace_total": trace_total,
        "by_depth": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_depth.items())},
    }


def load_data(args):
    root = Path(args.root)
    train_depths = _parse_ints(args.train_depths) or []
    test_depths = _parse_ints(args.test_depths) or []
    train_qdeps = _parse_ints(args.train_qdeps)
    test_qdeps = _parse_ints(args.test_qdeps)
    if args.dataset == "proofwriter":
        train = load_proofwriter(root, train_depths, "train", args.limit_train)
        tests = {f"depth-{d}": load_proofwriter(root, [d], "test", args.limit_test) for d in test_depths}
    elif args.dataset == "ruletaker_gfair":
        train = load_ruletaker_gfair(root, "train", args.limit_train)
        tests = {
            "dev": load_ruletaker_gfair(root, "dev", args.limit_test),
            "test": load_ruletaker_gfair(root, "test", args.limit_test),
        }
    elif args.dataset == "ruletaker_raw":
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
    return train, tests


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train, tests = load_data(args)
    model = MultiHopTruaProp(
        args.model_name,
        reasoner=args.reasoner,
        num_glimpses=args.num_glimpses,
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout,
    ).to(device)

    train_ds = TextTraceDataset(train, tokenizer, args.max_sents, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    train_log = []
    for epoch in range(args.epochs):
        model.train()
        totals = defaultdict(float)
        for batch in train_loader:
            batch = move(batch, device)
            logits, scores_list, attn_list = model(batch, coverage_penalty=args.coverage_penalty)
            main = nn.functional.cross_entropy(logits, batch["label"])
            step, cov, off = trace_losses(
                scores_list,
                attn_list,
                batch["mask"],
                batch["trace"],
                args.lambda_trace_coverage,
                args.lambda_offtrace,
            )
            loss = main + args.lambda_trace * step + cov + off
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            totals["loss"] += float(loss.item())
            totals["main"] += float(main.item())
            totals["trace_step"] += float(step.item())
            totals["trace_cov"] += float(cov.item())
            totals["offtrace"] += float(off.item())
        denom = max(len(train_loader), 1)
        row = {"epoch": epoch + 1, **{k: v / denom for k, v in sorted(totals.items())}}
        train_log.append(row)
        print(json.dumps(row), flush=True)

    results = {}
    for name, samples in tests.items():
        ds = TextTraceDataset(samples, tokenizer, args.max_sents, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate)
        results[name] = evaluate(model, loader, device, args.coverage_penalty)

    out = {
        "dataset": args.dataset,
        "reasoner": args.reasoner,
        "num_glimpses": args.num_glimpses,
        "model_name": args.model_name,
        "freeze_encoder": args.freeze_encoder,
        "lambda_trace": args.lambda_trace,
        "lambda_trace_coverage": args.lambda_trace_coverage,
        "lambda_offtrace": args.lambda_offtrace,
        "coverage_penalty": args.coverage_penalty,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_size": len(train),
        "train_log": train_log,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


def _parse_ints(value):
    if value is None or value == "":
        return None
    return [int(x) for x in str(value).split(",") if x != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["proofwriter", "ruletaker_gfair", "ruletaker_raw", "prontoqa"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-name", default="/vepfs/tsra_models/hf/bert-base-uncased")
    parser.add_argument("--reasoner", choices=["single", "multihop"], default="multihop")
    parser.add_argument("--num-glimpses", type=int, default=4)
    parser.add_argument("--lambda-trace", type=float, default=1.0)
    parser.add_argument("--lambda-trace-coverage", type=float, default=0.25)
    parser.add_argument("--lambda-offtrace", type=float, default=0.05)
    parser.add_argument("--coverage-penalty", type=float, default=0.2)
    parser.add_argument("--train-depths", default="0,1,2")
    parser.add_argument("--test-depths", default="3,5")
    parser.add_argument("--train-qdeps", default="")
    parser.add_argument("--test-qdeps", default="")
    parser.add_argument("--limit-train", type=int, default=3000)
    parser.add_argument("--limit-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-sents", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.limit_train is not None and args.limit_train <= 0:
        args.limit_train = None
    if args.limit_test is not None and args.limit_test <= 0:
        args.limit_test = None
    run(args)


if __name__ == "__main__":
    main()
