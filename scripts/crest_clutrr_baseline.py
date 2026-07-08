#!/usr/bin/env python3
"""CREST-style counterfactual CLUTRR baseline.

This is an independent, fair-input adaptation for the CREST shortcut-mitigation
idea when official code is not available. It trains a plain encoder classifier
with final-label supervision plus two counterfactual views:

- entity-renamed story/query with the same label;
- reversed query with the inverse kinship label.

It does not use gold traces, paths, entity graphs, or TRUA modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clutrr.cli.baseline import BaselineModel, evaluate
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.models.backbones import build_tokenizer, is_decoder_only_model
from clutrr.utils.parsing import parse_pair_literal
from clutrr.utils.seed import set_seed


MALE_NAMES = [
    "Aaron",
    "Brian",
    "Caleb",
    "David",
    "Ethan",
    "Frank",
    "George",
    "Henry",
    "Isaac",
    "Jack",
    "Kevin",
    "Liam",
]
FEMALE_NAMES = [
    "Alice",
    "Beth",
    "Clara",
    "Diana",
    "Eva",
    "Fiona",
    "Grace",
    "Helen",
    "Iris",
    "Julia",
    "Karen",
    "Laura",
]

GENDERED_GROUPS = {
    "father": ("male", "parent"),
    "mother": ("female", "parent"),
    "son": ("male", "child"),
    "daughter": ("female", "child"),
    "brother": ("male", "sibling"),
    "sister": ("female", "sibling"),
    "uncle": ("male", "pibling"),
    "aunt": ("female", "pibling"),
    "nephew": ("male", "nibling"),
    "niece": ("female", "nibling"),
    "grandfather": ("male", "grandparent"),
    "grandmother": ("female", "grandparent"),
    "grandson": ("male", "grandchild"),
    "granddaughter": ("female", "grandchild"),
    "husband": ("male", "spouse"),
    "wife": ("female", "spouse"),
    "father-in-law": ("male", "parent_in_law"),
    "mother-in-law": ("female", "parent_in_law"),
    "son-in-law": ("male", "child_in_law"),
    "daughter-in-law": ("female", "child_in_law"),
}

INVERSE_GROUP = {
    "parent": "child",
    "child": "parent",
    "sibling": "sibling",
    "pibling": "nibling",
    "nibling": "pibling",
    "grandparent": "grandchild",
    "grandchild": "grandparent",
    "spouse": "spouse",
    "parent_in_law": "child_in_law",
    "child_in_law": "parent_in_law",
}

RELATION_BY_GROUP_GENDER = {(group, gender): rel for rel, (gender, group) in GENDERED_GROUPS.items()}


def parse_genders(text: str) -> dict[str, str]:
    genders: dict[str, str] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        name, gender = item.split(":", 1)
        gender = gender.strip().lower()
        if gender in {"male", "female"}:
            genders[name.strip()] = gender
    return genders


def inverse_relation_for_reversed_query(relation: str, original_subject_gender: str | None) -> str | None:
    if relation not in GENDERED_GROUPS or original_subject_gender not in {"male", "female"}:
        return None
    _, group = GENDERED_GROUPS[relation]
    inv_group = INVERSE_GROUP[group]
    return RELATION_BY_GROUP_GENDER.get((inv_group, original_subject_gender))


def replace_name(text: str, old: str, new: str) -> str:
    text = text.replace(f"[{old}]", f"[{new}]")
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def renamed_view(story: str, query: tuple[str, str], genders: dict[str, str], rng: random.Random):
    names = sorted(genders)
    male_pool = MALE_NAMES[:]
    female_pool = FEMALE_NAMES[:]
    rng.shuffle(male_pool)
    rng.shuffle(female_pool)
    mapping: dict[str, str] = {}
    male_idx = 0
    female_idx = 0
    for name in names:
        if genders.get(name) == "male":
            replacement = male_pool[male_idx % len(male_pool)]
            male_idx += 1
        else:
            replacement = female_pool[female_idx % len(female_pool)]
            female_idx += 1
        if replacement == name:
            replacement = f"{replacement}X"
        mapping[name] = replacement

    new_story = story
    for old in sorted(mapping, key=len, reverse=True):
        new_story = replace_name(new_story, old, mapping[old])

    return new_story, (mapping.get(query[0], query[0]), mapping.get(query[1], query[1]))


class CrestCLUTRRDataset(Dataset):
    def __init__(self, root: str, dataset: str, split: str, data_percentage: int = 100, seed: int = 0):
        self.dataset_dir = os.path.join(root, dataset)
        self.seed = seed
        self.data = []

        if os.path.exists(self.dataset_dir):
            file_names = [
                os.path.join(self.dataset_dir, name)
                for name in os.listdir(self.dataset_dir)
                if f"_{split}.csv" in name
            ]
            for file_name in sorted(file_names):
                with open(file_name, "r", newline="") as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader)
                    self.data.extend(list(reader))
        else:
            print(f"Warning: Directory {self.dataset_dir} not found.")

        data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[:data_num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        story = row[2]
        query = parse_pair_literal(row[3])
        if query is None:
            raise ValueError(f"Invalid query tuple format: {row[3]}")
        relation = row[5]
        genders = parse_genders(row[14] if len(row) > 14 else "")

        try:
            hops = int(row[10].split(".")[-1])
        except Exception:
            hops = -1

        rng = random.Random((self.seed + 17) * 1000003 + index)
        renamed_story, renamed_query = renamed_view(story, query, genders, rng)

        reverse_relation = inverse_relation_for_reversed_query(relation, genders.get(query[0]))
        if reverse_relation is None:
            reverse_query = query
            reverse_label = relation_id_map.get(relation, relation_id_map["nothing"])
            reverse_valid = False
        else:
            reverse_query = (query[1], query[0])
            reverse_label = relation_id_map[reverse_relation]
            reverse_valid = True

        return {
            "story": story,
            "query": f"{query[0]} and {query[1]}",
            "label": relation_id_map.get(relation, relation_id_map["nothing"]),
            "renamed_story": renamed_story,
            "renamed_query": f"{renamed_query[0]} and {renamed_query[1]}",
            "reverse_story": story,
            "reverse_query": f"{reverse_query[0]} and {reverse_query[1]}",
            "reverse_label": reverse_label,
            "reverse_valid": reverse_valid,
            "hops": hops,
        }


class CrestCollator:
    def __init__(self, tokenizer, device: torch.device, model_type: str):
        self.tokenizer = tokenizer
        self.device = device
        self.decoder_only = is_decoder_only_model(model_type)

    @staticmethod
    def _decoder_prompt(story: str, query: str) -> str:
        return f"Story: {story}\nQuestion: What is the relation between {query}?"

    def encode(self, stories, queries):
        if self.decoder_only:
            prompts = [self._decoder_prompt(story, query) for story, query in zip(stories, queries)]
            return self.tokenizer(
                text=prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
        return self.tokenizer(
            text=stories,
            text_pair=queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

    def __call__(self, batch):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long, device=self.device)
        reverse_labels = torch.tensor([b["reverse_label"] for b in batch], dtype=torch.long, device=self.device)
        reverse_mask = torch.tensor([b["reverse_valid"] for b in batch], dtype=torch.bool, device=self.device)
        hops = torch.tensor([b["hops"] for b in batch], dtype=torch.long, device=self.device)

        base = self.encode([b["story"] for b in batch], [b["query"] for b in batch])
        renamed = self.encode([b["renamed_story"] for b in batch], [b["renamed_query"] for b in batch])
        reverse = self.encode([b["reverse_story"] for b in batch], [b["reverse_query"] for b in batch])

        return {
            "base_input_ids": base.input_ids,
            "base_attention_mask": base.attention_mask,
            "renamed_input_ids": renamed.input_ids,
            "renamed_attention_mask": renamed.attention_mask,
            "reverse_input_ids": reverse.input_ids,
            "reverse_attention_mask": reverse.attention_mask,
            "labels": labels,
            "reverse_labels": reverse_labels,
            "reverse_mask": reverse_mask,
            "hops": hops,
        }


def evaluate_crest(model, loader):
    model.eval()
    total = 0
    correct = 0
    hop_total: dict[int, int] = {}
    hop_correct: dict[int, int] = {}
    rename_consistent = 0
    reverse_valid_count = 0
    reverse_correct = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            hops = batch["hops"]
            out = model(batch["base_input_ids"], batch["base_attention_mask"])
            preds = torch.argmax(out["logits"], dim=1)

            renamed_out = model(batch["renamed_input_ids"], batch["renamed_attention_mask"])
            renamed_preds = torch.argmax(renamed_out["logits"], dim=1)
            rename_consistent += (preds == renamed_preds).sum().item()

            reverse_mask = batch["reverse_mask"]
            if reverse_mask.any():
                reverse_out = model(batch["reverse_input_ids"], batch["reverse_attention_mask"])
                reverse_preds = torch.argmax(reverse_out["logits"], dim=1)
                reverse_valid_count += reverse_mask.sum().item()
                reverse_correct += (reverse_preds[reverse_mask] == batch["reverse_labels"][reverse_mask]).sum().item()

            total += len(labels)
            correct += (preds == labels).sum().item()
            for hop, pred, target in zip(hops.cpu().numpy(), preds.cpu().numpy(), labels.cpu().numpy()):
                if hop == -1:
                    continue
                hop_total[int(hop)] = hop_total.get(int(hop), 0) + 1
                hop_correct[int(hop)] = hop_correct.get(int(hop), 0) + int(pred == target)

    short_correct = sum(hop_correct.get(h, 0) for h in [2, 3])
    short_total = sum(hop_total.get(h, 0) for h in [2, 3])
    long_correct = sum(hop_correct.get(h, 0) for h in range(6, 20))
    long_total = sum(hop_total.get(h, 0) for h in range(6, 20))
    per_hop = {
        str(h): {
            "accuracy": hop_correct.get(h, 0) / hop_total[h],
            "correct": hop_correct.get(h, 0),
            "total": hop_total[h],
        }
        for h in sorted(hop_total)
    }
    return {
        "overall": correct / total if total else 0.0,
        "short_hop": short_correct / short_total if short_total else 0.0,
        "long_hop": long_correct / long_total if long_total else 0.0,
        "rename_consistency": rename_consistent / total if total else 0.0,
        "reverse_cf_accuracy": reverse_correct / reverse_valid_count if reverse_valid_count else 0.0,
        "per_hop": per_hop,
        "total": total,
    }


def train(args):
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(args.model_type, model_name_or_path=args.model_name_or_path)
    collator = CrestCollator(tokenizer, device=device, model_type=args.model_type)

    train_ds = CrestCLUTRRDataset(args.root, args.dataset, "train", seed=args.seed)
    test_ds = CrestCLUTRRDataset(args.root, args.dataset, "test", seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    model = BaselineModel(args.model_type, len(relation_id_map), model_name_or_path=args.model_name_or_path)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    run_meta = {
        "method": "crest_style_query_reverse_rename",
        "official_code": False,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "cf_weight": args.cf_weight,
        "consistency_weight": args.consistency_weight,
        "uses_trace": False,
        "uses_gold_path": False,
        "test_time_input": "raw_story_query_only",
    }
    print(json.dumps({"run_meta": run_meta}, ensure_ascii=False))

    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_main = 0.0
        total_rename = 0.0
        total_reverse = 0.0
        total_kl = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            base = model(batch["base_input_ids"], batch["base_attention_mask"])
            renamed = model(batch["renamed_input_ids"], batch["renamed_attention_mask"])
            reverse = model(batch["reverse_input_ids"], batch["reverse_attention_mask"])

            main_loss = F.cross_entropy(base["logits"], batch["labels"])
            rename_loss = F.cross_entropy(renamed["logits"], batch["labels"])
            if batch["reverse_mask"].any():
                reverse_loss = F.cross_entropy(reverse["logits"][batch["reverse_mask"]], batch["reverse_labels"][batch["reverse_mask"]])
            else:
                reverse_loss = torch.zeros((), device=device)
            kl_loss = F.kl_div(
                F.log_softmax(renamed["logits"], dim=-1),
                F.softmax(base["logits"].detach(), dim=-1),
                reduction="batchmean",
            )
            loss = main_loss + args.cf_weight * 0.5 * (rename_loss + reverse_loss) + args.consistency_weight * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_main += main_loss.item()
            total_rename += rename_loss.item()
            total_reverse += reverse_loss.item()
            total_kl += kl_loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_stats = {
            "loss": total_loss / max(1, len(train_loader)),
            "main_loss": total_main / max(1, len(train_loader)),
            "rename_loss": total_rename / max(1, len(train_loader)),
            "reverse_loss": total_reverse / max(1, len(train_loader)),
            "kl_loss": total_kl / max(1, len(train_loader)),
        }
        metrics = evaluate_crest(model, test_loader)
        row = {"epoch": epoch, "train": train_stats, "eval": metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if best is None or metrics["overall"] > best["eval"]["overall"]:
            best = row

    out = {"run_meta": run_meta, "best": best, "final": history[-1] if history else None, "history": history}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "final": history[-1] if history else None}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--dataset", default="data_089907f8")
    parser.add_argument("--model_type", default="deberta-v3")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cf_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
