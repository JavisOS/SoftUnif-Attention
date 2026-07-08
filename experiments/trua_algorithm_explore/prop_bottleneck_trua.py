#!/usr/bin/env python3
"""Exploratory TRUA-v2 bottleneck runner for Prop-style reasoning tasks.

This script is isolated from the main TRUA implementation.  It tests whether
forcing the final answer through a trace-supervised evidence state improves the
trace-to-accuracy coupling that the current TRUA-Prop runner only weakly enforces.
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


class BottleneckTruaProp(nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        num_glimpses: int = 4,
        freeze_encoder: bool = False,
        text_residual_weight: float = 0.2,
        bypass_dropout: float = 0.5,
        coverage_penalty: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_glimpses = max(1, num_glimpses)
        self.text_residual_weight = text_residual_weight
        self.bypass_dropout = bypass_dropout
        self.coverage_penalty = coverage_penalty

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

        self.reason_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.text_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return out[:, 0]

    def _pack_sentences(self, flat_sent, sent_lens, max_s):
        bsz = sent_lens.size(0)
        hidden = flat_sent.size(-1)
        sent = torch.zeros(bsz, max_s, hidden, device=flat_sent.device, dtype=flat_sent.dtype)
        cursor = 0
        for i, n in enumerate(sent_lens.tolist()):
            sent[i, :n] = flat_sent[cursor : cursor + n]
            cursor += n
        return sent

    def forward(self, batch):
        text = self.encode_cls(batch["text_ids"], batch["text_mask"])
        query = self.query_proj(self.encode_cls(batch["query_ids"], batch["query_mask"]))
        flat_sent = self.sent_proj(self.encode_cls(batch["sent_ids"], batch["sent_mask"]))
        sent = self._pack_sentences(flat_sent, batch["sent_lens"], batch["mask"].size(1))

        hidden = sent.size(-1)
        state = self.state_norm(query)
        coverage = sent.new_zeros(batch["label"].size(0), batch["mask"].size(1))
        scores_list = []
        attn_list = []
        ctx_list = []
        for _ in range(self.num_glimpses):
            scores = (sent * state.unsqueeze(1)).sum(-1) / math.sqrt(hidden)
            if self.coverage_penalty > 0:
                scores = scores - self.coverage_penalty * coverage
            scores = scores.masked_fill(~batch["mask"], -1e4)
            attn = torch.softmax(scores, dim=-1)
            ctx = (attn.unsqueeze(-1) * sent).sum(1)
            state = self.state_norm(self.gru(ctx, state))
            coverage = coverage + attn
            scores_list.append(scores)
            attn_list.append(attn)
            ctx_list.append(ctx)

        ctx_stack = torch.stack(ctx_list, dim=1)
        hop_query = state.unsqueeze(1).expand_as(ctx_stack)
        hop_weights = torch.softmax(self.hop_gate(torch.cat([ctx_stack, hop_query], dim=-1)).squeeze(-1), dim=-1)
        ctx_weighted = (hop_weights.unsqueeze(-1) * ctx_stack).sum(1)
        ctx_max = ctx_stack.max(dim=1).values

        reason_logits = self.reason_head(torch.cat([state, ctx_weighted, ctx_max], dim=-1))
        text_logits = self.text_head(text)
        text_logits_for_fusion = text_logits
        if self.training and self.bypass_dropout > 0 and self.text_residual_weight != 0:
            keep = torch.rand(text_logits.size(0), 1, device=text_logits.device) >= self.bypass_dropout
            text_logits_for_fusion = text_logits * keep.to(text_logits.dtype)
        fused_logits = reason_logits + self.text_residual_weight * text_logits_for_fusion
        return {
            "fused_logits": fused_logits,
            "reason_logits": reason_logits,
            "text_logits": text_logits,
            "scores_list": scores_list,
            "attn_list": attn_list,
        }


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


def _acc_update(stats, key, pred, label):
    stats[key][0] += int((pred == label).sum().item())
    stats[key][1] += int(pred.numel())


def evaluate(model, loader, device):
    model.eval()
    stats = defaultdict(lambda: [0, 0])
    trace_top1 = trace_any = trace_total = 0
    gold_mass_sum = 0.0
    by_depth = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            out = model(batch)
            labels = batch["label"]
            fused_pred = out["fused_logits"].argmax(-1)
            reason_pred = out["reason_logits"].argmax(-1)
            text_pred = out["text_logits"].argmax(-1)
            _acc_update(stats, "fused", fused_pred, labels)
            _acc_update(stats, "reason", reason_pred, labels)
            _acc_update(stats, "text", text_pred, labels)

            top_each = [scores.masked_fill(~batch["mask"], -1e4).argmax(-1) for scores in out["scores_list"]]
            attn_stack = torch.stack(out["attn_list"], dim=1)
            for i in range(labels.size(0)):
                valid_trace = batch["trace"][i][batch["mask"][i]]
                if valid_trace.sum().item() <= 0:
                    continue
                trace_total += 1
                trace_top1 += int(batch["trace"][i, top_each[0][i]].item() > 0)
                trace_any += int(any(batch["trace"][i, top[i]].item() > 0 for top in top_each))
                gold_mass_sum += float((attn_stack[i] * batch["trace"][i].unsqueeze(0)).sum(dim=-1).mean().item())

            for d, p, y in zip(batch["depth"].cpu().tolist(), fused_pred.cpu().tolist(), labels.cpu().tolist()):
                by_depth[d][1] += 1
                by_depth[d][0] += int(p == y)

    return {
        "fused_accuracy": stats["fused"][0] / max(stats["fused"][1], 1),
        "reason_accuracy": stats["reason"][0] / max(stats["reason"][1], 1),
        "text_accuracy": stats["text"][0] / max(stats["text"][1], 1),
        "total": stats["fused"][1],
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
    model = BottleneckTruaProp(
        args.model_name,
        num_glimpses=args.num_glimpses,
        freeze_encoder=args.freeze_encoder,
        text_residual_weight=args.text_residual_weight,
        bypass_dropout=args.bypass_dropout,
        coverage_penalty=args.coverage_penalty,
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
            out = model(batch)
            labels = batch["label"]
            fused_ce = nn.functional.cross_entropy(out["fused_logits"], labels)
            reason_ce = nn.functional.cross_entropy(out["reason_logits"], labels)
            text_ce = nn.functional.cross_entropy(out["text_logits"], labels)
            step, cov, off = trace_losses(
                out["scores_list"],
                out["attn_list"],
                batch["mask"],
                batch["trace"],
                args.lambda_trace_coverage,
                args.lambda_offtrace,
            )
            loss = (
                fused_ce
                + args.lambda_reason_ce * reason_ce
                + args.lambda_text_ce * text_ce
                + args.lambda_trace * step
                + cov
                + off
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            totals["loss"] += float(loss.item())
            totals["fused_ce"] += float(fused_ce.item())
            totals["reason_ce"] += float(reason_ce.item())
            totals["text_ce"] += float(text_ce.item())
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
        results[name] = evaluate(model, loader, device)

    out = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "freeze_encoder": args.freeze_encoder,
        "num_glimpses": args.num_glimpses,
        "text_residual_weight": args.text_residual_weight,
        "bypass_dropout": args.bypass_dropout,
        "lambda_reason_ce": args.lambda_reason_ce,
        "lambda_text_ce": args.lambda_text_ce,
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
    parser.add_argument("--num-glimpses", type=int, default=4)
    parser.add_argument("--text-residual-weight", type=float, default=0.2)
    parser.add_argument("--bypass-dropout", type=float, default=0.5)
    parser.add_argument("--lambda-reason-ce", type=float, default=1.0)
    parser.add_argument("--lambda-text-ce", type=float, default=0.0)
    parser.add_argument("--lambda-trace", type=float, default=1.0)
    parser.add_argument("--lambda-trace-coverage", type=float, default=0.25)
    parser.add_argument("--lambda-offtrace", type=float, default=0.05)
    parser.add_argument("--coverage-penalty", type=float, default=0.2)
    parser.add_argument("--train-depths", default="0,1,2")
    parser.add_argument("--test-depths", default="3,5")
    parser.add_argument("--train-qdeps", default="")
    parser.add_argument("--test-qdeps", default="")
    parser.add_argument("--limit-train", type=int, default=5000)
    parser.add_argument("--limit-test", type=int, default=2000)
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
