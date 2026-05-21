#!/usr/bin/env python3
"""FaiRR rule/fact selector adapted DeBERTa component runner."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class FairrSelectorDataset(Dataset):
    def __init__(self, root: Path, split: str, limit: int | None = None):
        folder = root / split
        self.input_ids = load_pkl(folder / "input_ids.pkl")
        self.labels = load_pkl(folder / "token_labels.pkl")
        self.mask = load_pkl(folder / "token_mask.pkl")
        if limit:
            self.input_ids = self.input_ids[:limit]
            self.labels = self.labels[:limit]
            self.mask = self.mask[:limit]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.LongTensor(self.input_ids[idx]),
            "labels": torch.FloatTensor(self.labels[idx]),
            "token_mask": torch.BoolTensor(self.mask[idx]),
        }


class Collator:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, items):
        ids = pad_sequence([x["input_ids"] for x in items], batch_first=True, padding_value=self.pad_id)
        labels = pad_sequence([x["labels"] for x in items], batch_first=True, padding_value=0)
        mask = pad_sequence([x["token_mask"] for x in items], batch_first=True, padding_value=0).bool()
        return {"input_ids": ids, "attention_mask": (ids != self.pad_id).long(), "labels": labels, "token_mask": mask}


class DebertaTokenSelector(nn.Module):
    def __init__(self, model_name: str, freeze_encoder: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.classifier(h).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    top1 = total = 0
    token_correct = token_total = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["token_mask"].to(device)
            logits = model(ids, attn)
            pred_tokens = (logits > 0).float()
            token_correct += ((pred_tokens == labels).float() * mask.float()).sum().item()
            token_total += mask.sum().item()
            masked_logits = logits.masked_fill(~mask, -1e4)
            top = masked_logits.argmax(-1)
            for i, j in enumerate(top.cpu().tolist()):
                if labels[i][mask[i]].sum().item() > 0:
                    total += 1
                    top1 += int(labels[i, j].item() > 0)
    return {"top1_acc": top1 / max(total, 1), "token_acc": token_correct / max(token_total, 1), "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--model-name", default="microsoft/deberta-base")
    ap.add_argument("--train-limit", type=int, default=2000)
    ap.add_argument("--eval-limit", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.processed_root)
    train = FairrSelectorDataset(root, "train", args.train_limit)
    dev = FairrSelectorDataset(root, "dev", args.eval_limit)
    test = FairrSelectorDataset(root, "test", args.eval_limit)
    collate = Collator(pad_id=0)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaTokenSelector(args.model_name, args.freeze_encoder).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"FaiRR selector epoch {ep+1}"):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["token_mask"].to(device)
            logits = model(ids, attn)
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"epoch={ep+1} loss={loss_sum / max(len(train_loader), 1):.4f}")
    result = {
        "method": "FaiRR selector adapted DeBERTa",
        "processed_root": str(root),
        "train_examples": len(train),
        "dev": evaluate(model, dev_loader, device),
        "test": evaluate(model, test_loader, device),
        "note": "FaiRR selector data/objective with DeBERTa encoder; component result, not full FaiRR inference.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
