import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clutrr.models.backbones import build_backbone_model, build_tokenizer, is_decoder_only_model
from clutrr.utils.parsing import parse_pair_literal
from clutrr.config.defaults import DEFAULT_CLUTRR_DATASET, DEFAULT_CLUTRR_ROOT
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map
from clutrr.utils.seed import set_seed
from clutrr.training.model_selection import (
    clone_model_state,
    restore_model_state,
    stratified_train_validation_split,
    write_metrics,
)


BASELINE_DEFAULTS = {
    "model_type": "roberta",
    "model_name_or_path": None,
    "root": DEFAULT_CLUTRR_ROOT,
    "dataset": DEFAULT_CLUTRR_DATASET,
    "batch_size": 16,
    "eval_batch_size": 32,
    "num_workers": 0,
    "lr": 2e-5,
    "epochs": 10,
    "gpus": "0",
    "seed": 42,
    "use_qlora": False,
    "load_in_4bit": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "pooling": None,
    "validation_fraction": 0.1,
    "validation_seed": 2027,
    "metrics_out": None,
    "train_data_percentage": 100,
    "test_data_percentage": 100,
}


def _extract_config_path(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _load_yaml_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping/dict: {config_path}")
    return data


class CLUTRRBaselineDataset(Dataset):
    """Baseline dataset wrapper for CLUTRR CSV files."""

    def __init__(self, root, dataset, split, data_percentage=100):
        self.dataset_dir = os.path.join(root, f"{dataset}/")

        if os.path.exists(self.dataset_dir):
            self.file_names = sorted(
                os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d
            )
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
    def __init__(self, tokenizer, device, model_type="roberta"):
        self.tokenizer = tokenizer
        self.device = device
        self.decoder_only = is_decoder_only_model(model_type)

    @staticmethod
    def _format_decoder_input(story, query):
        return f"Story: {story}\nQuestion: What is the relation between {query}?"

    def __call__(self, batch):
        stories = [b["story"] for b in batch]
        queries = [b["query"] for b in batch]
        targets = torch.tensor([b["target_id"] for b in batch], dtype=torch.long).to(self.device)
        hops = torch.tensor([b["hops"] for b in batch], dtype=torch.long).to(self.device)

        if self.decoder_only:
            prompts = [self._format_decoder_input(story, query) for story, query in zip(stories, queries)]
            enc = self.tokenizer(
                text=prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
        else:
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
    def __init__(
        self,
        model_type,
        num_labels,
        *,
        model_name_or_path=None,
        use_qlora=False,
        load_in_4bit=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        pooling=None,
    ):
        super().__init__()
        model_type = model_type.lower()
        self.decoder_only = is_decoder_only_model(model_type)
        self.pooling = pooling or ("last_token" if self.decoder_only else "cls")
        self.encoder = build_backbone_model(
            model_type,
            model_name_or_path=model_name_or_path,
            use_qlora=use_qlora,
            load_in_4bit=load_in_4bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def _pool_sequence(self, output, attention_mask):
        if self.pooling == "cls":
            pooled = output[:, 0, :]
        else:
            token_lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(output.size(0), device=output.device)
            pooled = output[batch_indices, token_lengths, :]
        return pooled

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = self._pool_sequence(out.last_hidden_state, attention_mask)

        pooled = pooled.to(device=self.classifier.weight.device, dtype=self.classifier.weight.dtype)

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
    per_hop = {}
    for hop in sorted(hop_total):
        total_for_hop = hop_total[hop]
        correct_for_hop = hop_correct.get(hop, 0)
        per_hop[int(hop)] = {
            "accuracy": correct_for_hop / total_for_hop if total_for_hop > 0 else 0.0,
            "correct": int(correct_for_hop),
            "total": int(total_for_hop),
        }
    return overall_acc, short_acc, long_acc, per_hop


def build_arg_parser(defaults=None):
    defaults = defaults or BASELINE_DEFAULTS
    parser = argparse.ArgumentParser(
        prog="python -m clutrr.cli.baseline",
        description="Train transformer baseline on CLUTRR only.",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path. CLI args override YAML values.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=defaults["model_type"],
        choices=[
            "bert",
            "roberta",
            "roberta-large",
            "deberta",
            "deberta-v3",
            "deberta-v3-large",
            "modernbert",
            "gpt2",
            "llama3.2-1b",
            "llama3.2-3b",
            "qwen2.5-7b",
            "qwen3-0.6b",
            "qwen3-0.6b-base",
            "qwen3-1.7b",
            "qwen3-1.7b-base",
            "qwen3-8b",
            "qwen3-8b-base",
        ],
        help="Backbone model.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=defaults["model_name_or_path"],
        help="Optional local/remote model path overriding built-in model id.",
    )
    parser.add_argument("--root", type=str, default=defaults["root"], help="CLUTRR data root.")
    parser.add_argument("--dataset", type=str, default=defaults["dataset"], help="CLUTRR dataset folder.")
    parser.add_argument(
        "--train_data_percentage",
        type=int,
        default=defaults["train_data_percentage"],
        help="Percentage of the training CSV used; keep at 100 for reported experiments.",
    )
    parser.add_argument(
        "--test_data_percentage",
        type=int,
        default=defaults["test_data_percentage"],
        help="Percentage of each test CSV used; keep at 100 for reported experiments.",
    )
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval_batch_size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--gpus", type=str, default=defaults["gpus"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=defaults["validation_fraction"],
        help="Fraction of the training set reserved for checkpoint selection.",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=defaults["validation_seed"],
        help="Fixed seed for the train/validation partition; independent of the model seed.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=defaults["metrics_out"],
        help="Optional JSON path for validation history and the selected test result.",
    )
    parser.add_argument(
        "--use_qlora",
        action=argparse.BooleanOptionalAction,
        default=defaults["use_qlora"],
        help="Enable LoRA adapters for decoder-only models.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=defaults["load_in_4bit"],
        help="Load backbone in 4bit quantized mode.",
    )
    parser.add_argument("--lora_r", type=int, default=defaults["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=defaults["lora_alpha"])
    parser.add_argument("--lora_dropout", type=float, default=defaults["lora_dropout"])
    parser.add_argument(
        "--pooling",
        type=str,
        default=defaults["pooling"],
        choices=["cls", "last_token", None],
        help="Sequence pooling strategy. Default picks model-specific strategy.",
    )
    return parser


def parse_baseline_args():
    config_path = _extract_config_path()
    defaults = dict(BASELINE_DEFAULTS)

    yaml_config = _load_yaml_config(config_path)
    unknown_keys = sorted(set(yaml_config.keys()) - set(defaults.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}")
    defaults.update(yaml_config)

    parser = build_arg_parser(defaults=defaults)
    return parser.parse_args()


def run():
    args = parse_baseline_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_type}")
    print("Loading CLUTRR data...")

    tokenizer = build_tokenizer(args.model_type, model_name_or_path=args.model_name_or_path)
    collator = BaselineCollator(tokenizer, device, model_type=args.model_type)

    full_train_ds = CLUTRRBaselineDataset(
        args.root,
        args.dataset,
        "train",
        data_percentage=args.train_data_percentage,
    )
    train_items = [full_train_ds[index] for index in range(len(full_train_ds))]
    train_strata = [(item["hops"], item["target_id"]) for item in train_items]
    train_ds, validation_ds = stratified_train_validation_split(
        full_train_ds,
        train_strata,
        validation_fraction=args.validation_fraction,
        seed=args.validation_seed,
    )
    test_ds = CLUTRRBaselineDataset(
        args.root,
        args.dataset,
        "test",
        data_percentage=args.test_data_percentage,
    )
    print(
        f"Train/validation split: {len(train_ds)}/{len(validation_ds)} "
        f"(fraction={args.validation_fraction}, seed={args.validation_seed})"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    model = BaselineModel(
        args.model_type,
        num_labels=len(relation_id_map),
        model_name_or_path=args.model_name_or_path,
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        pooling=args.pooling,
    )
    if not getattr(model.encoder, "is_loaded_in_4bit", False):
        model = model.to(device)
    else:
        model.classifier = model.classifier.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting Training...")
    best_validation = -1.0
    best_epoch = -1
    best_state = None
    validation_history = []
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
        overall, short_h, long_h, per_hop = evaluate(model, validation_loader)
        validation_history.append(
            {"epoch": epoch + 1, "overall": overall, "short_hop": short_h, "long_hop": long_h, "per_hop": per_hop}
        )
        print(f"  Validation Overall: {overall:.4f}")
        if overall > best_validation:
            best_validation = overall
            best_epoch = epoch + 1
            best_state = clone_model_state(model)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    restore_model_state(model, best_state)
    overall, short_h, long_h, per_hop = evaluate(model, test_loader)
    print(f"Selected Validation Epoch: {best_epoch} (overall={best_validation:.4f})")
    print("--> Test Evaluation (checkpoint selected on validation only)")
    print(f"  Overall Acc (Base): {overall:.4f}")
    print(f"  Short Hop (2-3):    {short_h:.4f}")
    print(f"  Long Hop (>=6):     {long_h:.4f}")
    if per_hop:
        parts = []
        for hop in sorted(per_hop):
            item = per_hop[hop]
            parts.append(f"{hop}={item['accuracy']:.4f} ({item['correct']}/{item['total']})")
        print(f"  Per-Hop Acc:        {', '.join(parts)}")
    write_metrics(
        args.metrics_out,
        {
            "dataset": args.dataset,
            "model_type": args.model_type,
            "seed": args.seed,
            "train_size": len(train_ds),
            "validation_size": len(validation_ds),
            "test_size": len(test_ds),
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "selected_epoch": best_epoch,
            "selected_validation_overall": best_validation,
            "validation_history": validation_history,
            "test": {"overall": overall, "short_hop": short_h, "long_hop": long_h, "per_hop": per_hop},
        },
    )


if __name__ == "__main__":
    run()
