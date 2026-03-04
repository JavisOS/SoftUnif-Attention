import argparse
import csv
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clutrr.models.backbones import build_backbone_model, build_tokenizer
from clutrr.utils.parsing import parse_pair_literal
from clutrr.config.defaults import DEFAULT_CLUTRR_DATASET, DEFAULT_CLUTRR_ROOT
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.utils.seed import set_seed


class CLUTRRBaselineDataset(Dataset):
    """Baseline dataset wrapper for CLUTRR CSV files."""

    def __init__(self, root, dataset, split, data_percentage=100):
        self.dataset_dir = os.path.join(root, f"{dataset}/")

        if os.path.exists(self.dataset_dir):
            self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
            self.data = []
            for file_name in self.file_names:
                with open(file_name, "r") as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader)  # skip header
                    self.data.extend(list(reader))
        else:
            self.data = []
            print(f"Warning: Directory {self.dataset_dir} not found.")

        self.data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[:self.data_num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        story = row[2]
        query_tuple = parse_pair_literal(row[3])
        if query_tuple is None:
            raise ValueError(f"Invalid query tuple format: {row[3]}")
        target_rel = row[5]

        try:
            task_name = row[10]
            hops = int(task_name.split(".")[-1])
        except Exception:
            hops = -1

        target_id = relation_id_map.get(target_rel, relation_id_map["nothing"])

        return {
            "story": story,
            "query": f"{query_tuple[0]} and {query_tuple[1]}",
            "target_id": target_id,
            "hops": hops,
        }


class BaselineCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        stories = [b["story"] for b in batch]
        queries = [b["query"] for b in batch]
        targets = torch.tensor([b["target_id"] for b in batch], dtype=torch.long).to(self.device)
        hops = torch.tensor([b["hops"] for b in batch], dtype=torch.long).to(self.device)

        enc = self.tokenizer(
            text=stories,
            text_pair=queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": targets,
            "hops": hops,
        }


class BaselineModel(nn.Module):
    def __init__(self, model_type, num_labels):
        super().__init__()
        self.encoder = build_backbone_model(model_type)
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    hop_correct = {}
    hop_total = {}

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            hops = batch["hops"]
            out = model(batch["input_ids"], batch["attention_mask"])
            preds = torch.argmax(out["logits"], dim=1)

            total += len(labels)
            correct += (preds == labels).sum().item()

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            hops_np = hops.cpu().numpy()
            for hop, pred, target in zip(hops_np, preds_np, labels_np):
                if hop == -1:
                    continue
                if hop not in hop_total:
                    hop_total[hop] = 0
                    hop_correct[hop] = 0
                hop_total[hop] += 1
                if pred == target:
                    hop_correct[hop] += 1

    overall_acc = correct / total if total > 0 else 0.0
    short_correct = sum(hop_correct.get(h, 0) for h in [2, 3])
    short_total = sum(hop_total.get(h, 0) for h in [2, 3])
    short_acc = short_correct / short_total if short_total > 0 else 0.0

    long_correct = sum(hop_correct.get(h, 0) for h in range(6, 20))
    long_total = sum(hop_total.get(h, 0) for h in range(6, 20))
    long_acc = long_correct / long_total if long_total > 0 else 0.0
    return overall_acc, short_acc, long_acc


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="python -m comparison.other_paper_method.code.cli.baseline",
        description="Train transformer baseline on CLUTRR only.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="roberta",
        choices=["bert", "roberta", "roberta-large", "deberta", "deberta-v3", "deberta-v3-large", "modernbert"],
        help="Backbone model.",
    )
    parser.add_argument("--root", type=str, default=DEFAULT_CLUTRR_ROOT, help="CLUTRR data root.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_CLUTRR_DATASET, help="CLUTRR dataset folder.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    return parser


def run():
    args = build_arg_parser().parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_type}")
    print("Loading CLUTRR data...")

    tokenizer = build_tokenizer(args.model_type)
    collator = BaselineCollator(tokenizer, device)

    train_ds = CLUTRRBaselineDataset(args.root, args.dataset, "train")
    test_ds = CLUTRRBaselineDataset(args.root, args.dataset, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator)

    model = BaselineModel(args.model_type, num_labels=len(relation_id_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}")
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")
        overall, short_h, long_h = evaluate(model, test_loader)
        print(f"  Overall Acc (Base): {overall:.4f}")
        print(f"  Short Hop (2-3):    {short_h:.4f}")
        print(f"  Long Hop (>=6):     {long_h:.4f}")


if __name__ == "__main__":
    run()
