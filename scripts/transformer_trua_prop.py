#!/usr/bin/env python3
"""TRUA proposition adapter for ProofWriter, RuleTaker, and PrOntoQA."""

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
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from clutrr.models.relation_attention import RelationConditionedEntityAttention
from clutrr.models.trua_core import (
    PROPOSITION_EVIDENCE_ADAPTER,
    TransitionRegularizedUnitAttentionCore,
    masked_evidence_distribution_loss,
)
from clutrr.training.model_selection import clone_model_state, repository_revision, restore_model_state
from generic_trua_prop import load_prontoqa, load_proofwriter, load_ruletaker_gfair, load_ruletaker_raw


def select_query_anchor(query, shared_anchor, use_goal_guidance):
    if use_goal_guidance:
        return query
    return shared_anchor.unsqueeze(0).expand(query.size(0), -1)


def validation_selection_key(metrics):
    """Prefer answer accuracy, using evidence selection only to break ties."""
    return float(metrics["accuracy"]), float(metrics["evidence_at_1"])


def complete_input_limits(dataset, max_sentences, max_context_tokens, allow_truncation=False):
    if allow_truncation:
        return max_sentences, max_context_tokens
    minimum_sentences = {
        "proofwriter": 32,
        "ruletaker_raw": 32,
        "ruletaker_gfair": 32,
        "prontoqa": 24,
    }
    return max(max_sentences, minimum_sentences[dataset]), max(max_context_tokens, 512)


class TextEvidenceDataset(Dataset):
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
        evidence = (x["trace_labels"] or [0])[: self.max_sents]
        if len(evidence) < len(sentences):
            evidence = evidence + [0] * (len(sentences) - len(evidence))
        return {
            "text": x["context"] + " [SEP] " + x["query"],
            "query": x["query"],
            "sentences": sentences,
            "evidence": evidence,
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
        evidence = torch.zeros(len(batch), max_s, dtype=torch.float32)
        cursor = 0
        for i, x in enumerate(batch):
            n = sent_lens[i]
            mask[i, :n] = True
            evidence[i, :n] = torch.tensor(x["evidence"][:n], dtype=torch.float32)
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
            "evidence": evidence,
            "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
            "depth": torch.tensor([x["depth"] for x in batch], dtype=torch.long),
        }


class TransformerTruaProp(TransitionRegularizedUnitAttentionCore):
    def __init__(
        self,
        model_name: str,
        freeze_encoder: bool = True,
        dropout: float = 0.1,
        relation_channels: int = 8,
        use_relation_conditioning: bool = True,
        use_goal_guidance: bool = True,
        use_aggregation_branch: bool = True,
        use_step_branch: bool = True,
    ):
        super().__init__()
        self.adapter_spec = PROPOSITION_EVIDENCE_ADAPTER
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.use_goal_guidance = use_goal_guidance
        self.sent_proj = nn.Linear(hidden, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.shared_anchor = nn.Parameter(torch.zeros(hidden))
        self.unit_attn = RelationConditionedEntityAttention(
            hidden_size=hidden,
            num_relations=relation_channels,
            dropout=dropout,
            top_k=None,
            use_relation_conditioning=use_relation_conditioning,
            use_goal_guidance=use_goal_guidance,
            use_aggregation_branch=use_aggregation_branch,
            use_step_branch=use_step_branch,
        )
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

        # The no-goal ablation removes query information from unit selection,
        # while the global answer head still receives the original text/query.
        anchor = select_query_anchor(query, self.shared_anchor, self.use_goal_guidance)
        units = torch.cat([anchor.unsqueeze(1), sent], dim=1)
        unit_mask = torch.cat(
            [torch.ones(bsz, 1, device=batch["mask"].device, dtype=torch.bool), batch["mask"]],
            dim=1,
        )
        updated, transitions = self.unit_attn(units, unit_mask, goal_embedding=query)
        anchor_edges = transitions["edge_index"][:, 0, :]
        anchor_logits = transitions["hop_logits"][:, 0, :]
        aligned_logits = anchor_logits.new_full((bsz, max_s + 1), -1e4)
        aligned_logits.scatter_(1, anchor_edges, anchor_logits)
        scores = aligned_logits[:, 1:].masked_fill(~batch["mask"], -1e4)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * updated[:, 1:]).sum(1)
        logits = self.classifier(torch.cat([text, ctx], dim=-1))
        return logits, scores


def move(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def evaluate(model, loader, device):
    model.eval()
    correct = total = evidence_hit = evidence_total = 0
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
                valid_evidence = batch["evidence"][i][batch["mask"][i]]
                if valid_evidence.sum().item() > 0:
                    evidence_total += 1
                    evidence_hit += int(batch["evidence"][i, j].item() > 0)
            for d, p, y in zip(batch["depth"].cpu().tolist(), pred.cpu().tolist(), batch["label"].cpu().tolist()):
                by_depth[d][1] += 1
                by_depth[d][0] += int(p == y)
    return {
        "accuracy": correct / max(total, 1),
        "total": total,
        "evidence_at_1": evidence_hit / max(evidence_total, 1),
        "evidence_total": evidence_total,
        "by_depth": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_depth.items())},
    }


def evidence_ce_loss(scores, mask, evidence):
    return masked_evidence_distribution_loss(scores, mask, evidence)


def run(train, validation, tests, args):
    code_revision = repository_revision()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = TransformerTruaProp(
        args.model_name,
        freeze_encoder=args.freeze_encoder,
        relation_channels=args.relation_channels,
        use_relation_conditioning=args.use_relation_conditioning,
        use_goal_guidance=args.use_goal_guidance,
        use_aggregation_branch=args.use_aggregation_branch,
        use_step_branch=args.use_step_branch,
    ).to(device)
    train_ds = TextEvidenceDataset(train, tokenizer, args.max_sents, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=train_ds.collate)
    validation_ds = TextEvidenceDataset(validation, tokenizer, args.max_sents, args.max_len)
    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=validation_ds.collate,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    best_validation_key = (-1.0, -1.0)
    best_epoch = -1
    best_state = None
    validation_history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            logits, scores = model(batch)
            loss = nn.functional.cross_entropy(logits, batch["label"])
            if args.lambda_evidence > 0:
                loss = loss + args.lambda_evidence * evidence_ce_loss(
                    scores,
                    batch["mask"],
                    batch["evidence"],
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        validation_metrics = evaluate(model, validation_loader, device)
        validation_history.append({"epoch": epoch + 1, **validation_metrics})
        print(
            f"epoch={epoch+1} loss={total_loss / max(len(train_loader), 1):.4f} "
            f"validation_acc={validation_metrics['accuracy']:.4f}"
        )
        validation_key = validation_selection_key(validation_metrics)
        if validation_key > best_validation_key:
            best_validation_key = validation_key
            best_epoch = epoch + 1
            best_state = clone_model_state(model)
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    restore_model_state(model, best_state)
    results = {}
    for name, samples in tests.items():
        ds = TextEvidenceDataset(samples, tokenizer, args.max_sents, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate)
        results[name] = evaluate(model, loader, device)
    return {
        "code_revision": code_revision,
        "dataset": args.dataset,
        "model_name": args.model_name,
        "lambda_evidence": args.lambda_evidence,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_depths": args.train_depths,
        "test_depths": args.test_depths,
        "train_qdeps": args.train_qdeps,
        "test_qdeps": args.test_qdeps,
        "max_sentences": args.max_sents,
        "max_context_tokens": args.max_len,
        "input_truncation_allowed": args.allow_input_truncation,
        "train": len(train),
        "validation": len(validation),
        "selected_epoch": best_epoch,
        "selected_validation_accuracy": best_validation_key[0],
        "selected_validation_evidence_at_1": best_validation_key[1],
        "selection_rule": "validation accuracy; Evidence@1 breaks exact ties",
        "validation_history": validation_history,
        "architecture": {
            "relation_channels": args.relation_channels,
            "use_relation_conditioning": args.use_relation_conditioning,
            "use_goal_guidance": args.use_goal_guidance,
            "use_aggregation_branch": args.use_aggregation_branch,
            "use_step_branch": args.use_step_branch,
        },
        "results": results,
    }


def _parse_ints(value):
    if value is None or value == "":
        return None
    return [int(x) for x in str(value).split(",") if x != ""]


def _split_samples(samples, validation_fraction, seed):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, round(len(indices) * validation_fraction))
    validation_size = min(validation_size, len(indices) - 1)
    validation_indices = set(indices[:validation_size])
    train = [sample for index, sample in enumerate(samples) if index not in validation_indices]
    validation = [sample for index, sample in enumerate(samples) if index in validation_indices]
    return train, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["proofwriter", "ruletaker_gfair", "ruletaker_raw", "prontoqa"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-name", default="microsoft/deberta-base")
    parser.add_argument(
        "--lambda-evidence",
        "--lambda-trace",
        dest="lambda_evidence",
        type=float,
        default=1.0,
    )
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
    parser.add_argument("--allow-input-truncation", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--relation-channels", type=int, default=8)
    parser.add_argument("--use-relation-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-goal-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-aggregation-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-step-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=2027)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.max_sents, args.max_len = complete_input_limits(
        args.dataset,
        args.max_sents,
        args.max_len,
        args.allow_input_truncation,
    )
    if args.limit_train is not None and args.limit_train <= 0:
        args.limit_train = None
    if args.limit_test is not None and args.limit_test <= 0:
        args.limit_test = None

    root = Path(args.root)
    if args.dataset == "proofwriter":
        train_depths = _parse_ints(args.train_depths) or []
        test_depths = _parse_ints(args.test_depths) or []
        train = load_proofwriter(root, train_depths, "train", args.limit_train)
        validation = load_proofwriter(root, train_depths, "dev", args.limit_test)
        tests = {f"depth-{d}": load_proofwriter(root, [d], "test", args.limit_test) for d in test_depths}
    elif args.dataset == "ruletaker_gfair":
        train = load_ruletaker_gfair(root, "train", args.limit_train)
        validation = load_ruletaker_gfair(root, "dev", args.limit_test)
        tests = {
            "test": load_ruletaker_gfair(root, "test", args.limit_test),
        }
    elif args.dataset == "ruletaker_raw":
        train_depths = _parse_ints(args.train_depths) or []
        test_depths = _parse_ints(args.test_depths) or []
        train_qdeps = _parse_ints(args.train_qdeps)
        test_qdeps = _parse_ints(args.test_qdeps)
        train = load_ruletaker_raw(root, train_depths, "train", args.limit_train, qdeps=train_qdeps)
        validation = load_ruletaker_raw(root, train_depths, "dev", args.limit_test, qdeps=train_qdeps)
        tests = {
            "test": load_ruletaker_raw(root, test_depths, "test", args.limit_test, qdeps=test_qdeps),
        }
    else:
        train_files = ["1hop_ProofsOnly_random_noadj.json", "2hop_ProofsOnly_random_noadj.json"]
        test_files = ["3hop_ProofsOnly_random_noadj.json", "4hop_ProofsOnly_random_noadj.json", "4hop_OOD_Composed_random_noadj.json"]
        train = load_prontoqa(root, train_files, args.limit_train)
        train, validation = _split_samples(train, args.validation_fraction, args.validation_seed)
        tests = {"ood": load_prontoqa(root, test_files, args.limit_test)}
    if not validation:
        train, validation = _split_samples(train, args.validation_fraction, args.validation_seed)
    result = run(train, validation, tests, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
