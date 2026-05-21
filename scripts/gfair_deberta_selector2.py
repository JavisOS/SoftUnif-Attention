#!/usr/bin/env python3
"""GFaiR Selector2 adapted to a local DeBERTa encoder.

This is a stronger runnable baseline than the tiny random-XLNet smoke test:
it uses GFaiR's RuleTaker mid-proof data construction and trains a
post-selector objective with a real local DeBERTa encoder. It is still marked
as an adapted GFaiR component, not the full T5/XLNet GFaiR pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def build_features(gfair_root: Path, data_dir: Path, split: str, tokenizer, limit: int, max_len: int):
    sys.path.insert(0, str(gfair_root / "data"))
    from create_my_example import createdata1

    rows = load_pickle(data_dir / f"{split}_withmidprove.pkl")
    features = []
    for row in tqdm(rows, desc=f"GFaiR build {split}"):
        nl_items, fol_items = [], []
        for mid_prove in row.get("gold_mid_proves", []):
            if len(mid_prove) == 2:
                nl_items.append(mid_prove[1][0].strip(".") + ".")
                fol_items.append(mid_prove[1][1])
            elif len(mid_prove) == 4 and nl_items:
                r1_num, r2_num, _, new_r = mid_prove
                try:
                    feat = createdata1(nl_items[:], fol_items[:], r1_num, r2_num, tokenizer)
                except Exception:
                    nl_items.append(new_r[0].strip(".") + ".")
                    fol_items.append(new_r[1])
                    continue
                if 0 < len(feat.tokens) <= max_len and feat.tobeselected:
                    features.append(feat)
                    if limit and len(features) >= limit:
                        return features
                nl_items.append(new_r[0].strip(".") + ".")
                fol_items.append(new_r[1])
    return features


class SelectorDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        f = self.features[i]
        return {
            "input_ids": torch.LongTensor(f.tokens),
            "labels": torch.FloatTensor(f.output),
            "token_mask": torch.BoolTensor(f.token_mask),
            "tobeselected": f.tobeselected,
            "positives": set(int(x) for x in f.positives),
        }


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, items):
        ids = pad_sequence([x["input_ids"] for x in items], batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence([x["labels"] for x in items], batch_first=True, padding_value=0)
        token_mask = pad_sequence([x["token_mask"] for x in items], batch_first=True, padding_value=0).bool()
        return {
            "input_ids": ids,
            "attention_mask": (ids != self.pad_id).long(),
            "labels": labels,
            "token_mask": token_mask,
            "tobeselected": [x["tobeselected"] for x in items],
            "positives": [x["positives"] for x in items],
        }


class DebertaSelector(nn.Module):
    def __init__(self, model_name: str, freeze_encoder: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(h).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    top1 = valid1 = total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids, attn).cpu()
            for i, candidates in enumerate(batch["tobeselected"]):
                if not candidates:
                    continue
                cand_scores = torch.tensor([logits[i, j].item() for j in candidates])
                best_local = int(cand_scores.argmax().item())
                best_tok = int(candidates[best_local])
                top1 += int(labels.cpu()[i, best_tok].item() > 0)
                valid1 += int(best_tok in batch["positives"][i])
                total += 1
    return {"top1_acc": top1 / max(total, 1), "top1_valid": valid1 / max(total, 1), "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfair-root", default="external_baselines/GFaiR")
    ap.add_argument("--data-dir", default="external_baselines/GFaiR/data/ruletaker_3ext_sat")
    ap.add_argument("--model-name", default="microsoft/deberta-base")
    ap.add_argument("--train-limit", type=int, default=1000)
    ap.add_argument("--eval-limit", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--out", default="outputs/external_baselines/gfair_deberta_selector2.json")
    args = ap.parse_args()
    if args.train_limit is not None and args.train_limit <= 0:
        args.train_limit = None
    if args.eval_limit is not None and args.eval_limit <= 0:
        args.eval_limit = None

    root = Path(args.gfair_root)
    data = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train = build_features(root, data, "train", tokenizer, args.train_limit, args.max_len)
    dev = build_features(root, data, "dev", tokenizer, args.eval_limit, args.max_len)
    test = build_features(root, data, "test", tokenizer, args.eval_limit, args.max_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaSelector(args.model_name, args.freeze_encoder).to(device)
    collate = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(SelectorDataset(train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(SelectorDataset(dev), batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(SelectorDataset(test), batch_size=args.batch_size, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        for batch in tqdm(train_loader, desc=f"GFaiR DeBERTa epoch {epoch+1}"):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["token_mask"].to(device)
            logits = model(ids, attn)
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += loss.item()
        print(f"epoch={epoch+1} loss={loss_total / max(len(train_loader), 1):.4f}")

    result = {
        "method": "GFaiR Selector2 adapted DeBERTa",
        "dataset": "RuleTaker / ruletaker_3ext_sat",
        "train_examples": len(train),
        "dev": evaluate(model, dev_loader, device),
        "test": evaluate(model, test_loader, device),
        "note": "GFaiR post-selector data/objective with DeBERTa encoder; not full T5+XLNet GFaiR pipeline.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
