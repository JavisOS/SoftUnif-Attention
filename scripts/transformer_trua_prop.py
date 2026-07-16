#!/usr/bin/env python3
"""TRUA proposition adapter for ProofWriter, RuleTaker, and PrOntoQA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import tokenizers
import transformers
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
from clutrr.models.relation_attention import RelationConditionedEntityAttention
from clutrr.models.controlled_unit_baselines import ContentSelfAttentionCore
from clutrr.models.trua_core import (
    PROPOSITION_EVIDENCE_ADAPTER,
    TransitionRegularizedUnitAttentionCore,
    masked_evidence_distribution_loss,
)
from clutrr.training.model_selection import clone_model_state, repository_revision, restore_model_state
from generic_trua_prop import (
    BINARY_LABEL_NAMES,
    PROOFWRITER_LABEL_NAMES,
    load_prontoqa,
    load_proofwriter,
    load_ruletaker_gfair,
    load_ruletaker_raw,
)


def select_query_anchor(query, shared_anchor, use_query_anchor):
    if use_query_anchor:
        return query
    return shared_anchor.unsqueeze(0).expand(query.size(0), -1)


def validation_selection_key(metrics):
    """Prefer answer accuracy, using evidence selection only to break ties."""
    evidence_at_1 = metrics["evidence_at_1"]
    return float(metrics["accuracy"]), 0.0 if evidence_at_1 is None else float(evidence_at_1)


def sample_fingerprint(samples):
    """Fingerprint the exact ordered examples used by an experiment split."""
    digest = hashlib.sha256()
    for sample in samples:
        record = {
            "id": sample.get("id"),
            "context": sample.get("context"),
            "query": sample.get("query"),
            "label": sample.get("label"),
            "depth": sample.get("depth"),
            "trace_labels": sample.get("trace_labels"),
        }
        digest.update(
            json.dumps(record, sort_keys=True, ensure_ascii=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def split_record(samples):
    examples = len(samples)
    evidence_examples = sum(bool(sum(sample.get("trace_labels") or [])) for sample in samples)
    label_counts = Counter(int(sample.get("label", 0)) for sample in samples)
    is_binary = set(label_counts).issubset({0, 1})
    return {
        "examples": examples,
        "sha256": sample_fingerprint(samples),
        "label_counts": {
            str(label): count for label, count in sorted(label_counts.items())
        },
        "positive_fraction": (
            label_counts[1] / max(examples, 1) if is_binary else None
        ),
        "evidence_fraction": evidence_examples / max(examples, 1),
    }


def label_names_for_dataset(dataset):
    if dataset == "proofwriter":
        return PROOFWRITER_LABEL_NAMES
    return BINARY_LABEL_NAMES


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
            "context": x["context"],
            "query": x["query"],
            "sentences": sentences,
            "evidence": evidence,
            "label": int(x.get("label", 1)),
            "depth": int(x.get("depth", -1)),
        }

    def collate(self, batch):
        tok = self.tokenizer
        texts = tok(
            [x["context"] for x in batch],
            [x["query"] for x in batch],
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
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
        result = {
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
        if "token_type_ids" in texts:
            result["text_token_type_ids"] = texts["token_type_ids"]
        return result


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
        use_query_anchor: bool | None = None,
        core_type: str = "trua",
        num_labels: int = 2,
    ):
        super().__init__()
        if core_type not in {"encoder", "trua", "self_attention"}:
            raise ValueError(f"Unsupported proposition core: {core_type}")
        self.adapter_spec = PROPOSITION_EVIDENCE_ADAPTER
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.use_goal_guidance = use_goal_guidance if core_type != "encoder" else False
        self.use_query_anchor = (
            use_goal_guidance if use_query_anchor is None else use_query_anchor
        ) if core_type != "encoder" else False
        self.core_type = core_type
        self.use_relation_conditioning = (
            use_relation_conditioning if core_type == "trua" else False
        )
        self.use_aggregation_branch = (
            use_aggregation_branch if core_type == "trua" else False
        )
        self.use_step_branch = use_step_branch if core_type == "trua" else False
        if core_type != "encoder":
            self.sent_proj = nn.Linear(hidden, hidden)
            self.query_proj = nn.Linear(hidden, hidden)
            self.shared_anchor = nn.Parameter(torch.zeros(hidden))
        if core_type == "trua":
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
        elif core_type == "self_attention":
            self.unit_attn = ContentSelfAttentionCore(hidden, num_heads=8, dropout=dropout)
        else:
            self.unit_attn = None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden if core_type == "encoder" else hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def encode_cls(self, input_ids, attention_mask, token_type_ids=None):
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        out = self.encoder(**encoder_inputs).last_hidden_state
        return out[:, 0]

    def forward(self, batch):
        text = self.encode_cls(
            batch["text_ids"],
            batch["text_mask"],
            batch.get("text_token_type_ids"),
        )
        if self.core_type == "encoder":
            return self.classifier(text), None

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

        # Anchor identity and explicit goal injection are independent controls.
        anchor = select_query_anchor(query, self.shared_anchor, self.use_query_anchor)
        units = torch.cat([anchor.unsqueeze(1), sent], dim=1)
        unit_mask = torch.cat(
            [torch.ones(bsz, 1, device=batch["mask"].device, dtype=torch.bool), batch["mask"]],
            dim=1,
        )
        if self.core_type == "trua":
            updated, transitions = self.unit_attn(units, unit_mask, goal_embedding=query)
            anchor_edges = transitions["edge_index"][:, 0, :]
            anchor_logits = transitions["hop_logits"][:, 0, :]
            aligned_logits = anchor_logits.new_full((bsz, max_s + 1), -1e4)
            aligned_logits.scatter_(1, anchor_edges, anchor_logits)
            scores = aligned_logits[:, 1:].masked_fill(~batch["mask"], -1e4)
        else:
            updated, attention_logits = self.unit_attn.forward_with_scores(
                units,
                unit_mask,
                query,
            )
            scores = attention_logits[:, 0, 1:].masked_fill(~batch["mask"], -1e4)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * updated[:, 1:]).sum(1)
        logits = self.classifier(torch.cat([text, ctx], dim=-1))
        return logits, scores


def move(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def evaluate(model, loader, device, label_names):
    model.eval()
    correct = total = evidence_hit = evidence_total = 0
    by_depth = defaultdict(lambda: [0, 0])
    by_label = defaultdict(lambda: [0, 0])
    confusion = torch.zeros(len(label_names), len(label_names), dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            logits, scores = model(batch)
            pred = logits.argmax(-1)
            correct += (pred == batch["label"]).sum().item()
            total += pred.numel()
            for p, y in zip(pred.cpu().tolist(), batch["label"].cpu().tolist()):
                by_label[y][1] += 1
                by_label[y][0] += int(p == y)
                confusion[y, p] += 1
            if scores is not None:
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
        "evidence_at_1": evidence_hit / evidence_total if evidence_total else None,
        "evidence_total": evidence_total,
        "by_depth": {str(k): v[0] / max(v[1], 1) for k, v in sorted(by_depth.items())},
        "by_depth_counts": {
            str(k): {"correct": v[0], "total": v[1]}
            for k, v in sorted(by_depth.items())
        },
        "by_label": {
            label_names[label]: by_label[label][0] / max(by_label[label][1], 1)
            for label in range(len(label_names))
        },
        "by_label_counts": {
            label_names[label]: {
                "correct": by_label[label][0],
                "total": by_label[label][1],
            }
            for label in range(len(label_names))
        },
        "macro_label_accuracy": sum(
            by_label[label][0] / max(by_label[label][1], 1)
            for label in range(len(label_names))
        ) / len(label_names),
        "confusion_matrix": confusion.tolist(),
    }


def evidence_ce_loss(scores, mask, evidence):
    return masked_evidence_distribution_loss(scores, mask, evidence)


def run(train, validation, tests, args):
    code_revision = repository_revision()
    label_names = label_names_for_dataset(args.dataset)
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
        use_query_anchor=args.use_query_anchor,
        core_type=args.core_type,
        num_labels=len(label_names),
    ).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    split_records = {
        "train": split_record(train),
        "validation": split_record(validation),
        "tests": {name: split_record(samples) for name, samples in tests.items()},
    }
    train_ds = TextEvidenceDataset(train, tokenizer, args.max_sents, args.max_len)
    data_order_seed = args.seed + 271828
    data_order_generator = torch.Generator()
    data_order_generator.manual_seed(data_order_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate,
        generator=data_order_generator,
    )
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
    optimization_seed = args.seed + 314159
    torch.manual_seed(optimization_seed)
    torch.cuda.manual_seed_all(optimization_seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started_at = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            logits, scores = model(batch)
            loss = nn.functional.cross_entropy(logits, batch["label"])
            if args.lambda_evidence > 0:
                if scores is None:
                    raise ValueError("Evidence regularization requires a unit-attention core")
                loss = loss + args.lambda_evidence * evidence_ce_loss(
                    scores,
                    batch["mask"],
                    batch["evidence"],
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        validation_metrics = evaluate(model, validation_loader, device, label_names)
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started_at
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    restore_model_state(model, best_state)
    results = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_started_at = time.perf_counter()
    for name, samples in tests.items():
        ds = TextEvidenceDataset(samples, tokenizer, args.max_sents, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=ds.collate)
        results[name] = evaluate(model, loader, device, label_names)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_started_at
    peak_cuda_memory_gib = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    return {
        "code_revision": code_revision,
        "dataset": args.dataset,
        "task_labels": {
            "num_labels": len(label_names),
            "id_to_name": {
                str(index): name for index, name in enumerate(label_names)
            },
        },
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
        "context_query_encoding": "tokenizer-native sequence pair",
        "input_truncation_allowed": args.allow_input_truncation,
        "train": len(train),
        "validation": len(validation),
        "split_records": split_records,
        "selected_epoch": best_epoch,
        "selected_validation_accuracy": best_validation_key[0],
        "selected_validation_evidence_at_1": best_validation_key[1],
        "selection_rule": "validation accuracy; Evidence@1 breaks exact ties",
        "resources": {
            "device": str(device),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "training_seconds": training_seconds,
            "test_seconds": test_seconds,
            "training_examples_per_second": (
                len(train) * args.epochs / max(training_seconds, 1e-9)
            ),
            "peak_cuda_memory_gib": peak_cuda_memory_gib,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "validation_history": validation_history,
        "architecture": {
            "core_type": args.core_type,
            "relation_channels": args.relation_channels,
            "use_relation_conditioning": model.use_relation_conditioning,
            "use_goal_guidance": model.use_goal_guidance,
            "use_query_anchor": model.use_query_anchor,
            "use_aggregation_branch": model.use_aggregation_branch,
            "use_step_branch": model.use_step_branch,
        },
        "randomness": {
            "model_initialization_seed": args.seed,
            "data_order_seed": data_order_seed,
            "optimization_seed": optimization_seed,
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
    parser.add_argument(
        "--core-type",
        choices=["encoder", "trua", "self_attention"],
        default="trua",
    )
    parser.add_argument("--use-relation-conditioning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-goal-guidance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-query-anchor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-aggregation-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-step-branch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=2027)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.core_type == "encoder" and args.lambda_evidence != 0.0:
        parser.error("--core-type encoder requires --lambda-evidence 0")
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
