import os
import argparse
import random
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset

from ruletaker.utils.distributed import is_main_process as _is_main_process
from ruletaker.models.backbones import build_ruletaker_encoder, build_ruletaker_tokenizer
from ruletaker.config.defaults import DEFAULT_RULETAKER_ROOT


RULETAKER_TRAIN_DEFAULTS = {
    "model_type": "roberta",
    "epochs": 10,
    "batch_size": 32,
    "lr": 2e-5,
    "root": DEFAULT_RULETAKER_ROOT,
    "gpus": "0",
    "limit": None,
    "train_depth": "depth-1,depth-2",
    "eval_depths": "depth-0,depth-1,depth-2,depth-3,depth-5",
    "eval_on_train": False,
    "max_length": 512,
    "seed": 42,
}


def _make_train_pbar(iterable, desc: str):
    # Avoid corrupted multi-line bars in non-TTY logs and on terminal resize.
    is_tty = sys.stderr.isatty()
    return tqdm(
        iterable,
        desc=desc,
        disable=not is_tty,
        dynamic_ncols=False,
        ncols=100,
        leave=False,
    )


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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. New RuleTaker Dataset Wrapper
# ==========================================
class RuleTakerDataset(Dataset):
    def __init__(self, split="train", limit=None, depth="all", dataset_dir=DEFAULT_RULETAKER_ROOT):
        import glob
        import json
        
        self.data = []
        
        # 1. Try Local Load
        # Pattern: data/rule-reasoning-dataset-V2020.2.5.0/original/depth-*/{split}.jsonl
        # or simplified path
        local_dir = dataset_dir
        files = []
        if os.path.exists(local_dir):
            if _is_main_process(): print(f"Looking for local RuleTaker in {local_dir}...")
            # Look in depth subfolders
            depth_list = [d.strip() for d in depth.split(",")] if depth and depth != "all" else ["depth-*"]
            for d in depth_list:
                files.extend(glob.glob(os.path.join(local_dir, d, f"{split}.jsonl")))
            files = sorted(files)
            if not files:
                 # Try direct
                 files = sorted(glob.glob(os.path.join(local_dir, f"{split}.jsonl")))
            
        if files:
            if _is_main_process(): print(f"Found {len(files)} local files. Loading...")
            for fpath in files:
                with open(fpath, 'r') as f:
                    for line in f:
                        obj = json.loads(line)
                        context = obj['context']
                        for q in obj['questions']:
                            self.data.append({
                                'context': context, 
                                'question': q['text'],
                                'label': 1 if q['label'] is True else 0
                            })
        else:
            # 2. Fallback to HF
            if _is_main_process(): print(f"Local files not found, downloading from HuggingFace...")
            hf_split = "validation" if split == "dev" else split
            ds = load_dataset("tasksource/ruletaker", split=hf_split)
            
            # Convert HF dataset to list of dicts for unified interface
            for item in ds:
                raw_label = item['label']
                if isinstance(raw_label, str):
                    label = 1 if raw_label == 'entailment' else 0
                else:
                    label = int(raw_label)
                    
                self.data.append({
                    'context': item.get('context', item.get('text', '')),
                    'question': item.get('question', ''),
                    'label': label
                })

        if limit and limit < len(self.data):
             self.data = self.data[:limit]

        if len(self.data) > 0 and _is_main_process():
            print(f"[{split}] Loaded {len(self.data)} samples.")
            print(f"[{split}] Example: {self.data[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class RuleTakerCollator:
    def __init__(self, tokenizer, device, max_length=512):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
    def __call__(self, batch):
        if not batch: return None
        
        contexts = [b['context'] for b in batch]
        questions = [b['question'] for b in batch]
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long).to(self.device)
        
        # Tokenize (Context + Question)
        enc = self.tokenizer(
            contexts,
            questions,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Diagnostic print (once)
        if not hasattr(self, 'debug_printed'):
             print("\n[DEBUG Collator] First batch input example (decoded):")
             print(self.tokenizer.decode(enc.input_ids[0]))
             print(f"[DEBUG Collator] Label: {labels[0].item()}")
             self.debug_printed = True
        
        return {
            'input_ids': enc.input_ids,
            'attention_mask': enc.attention_mask,
            'labels': labels
        }

# ==========================================
# 2. Simplified Model (CLS only)
# ==========================================
class RuleTakerClassifier(nn.Module):
    def __init__(self, device, tokenizer, model_type="roberta", num_labels=2):
        super().__init__()
        self.device = device
        self.model_type = model_type.lower()
        self.encoder = build_ruletaker_encoder(self.model_type)

        self.hidden_size = self.encoder.config.hidden_size
        
        # Simple Classifier Head (CLS)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(self, batch_data):
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        
        # Encoder
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = out.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        
        # Debug: Print logits stats once
        if not hasattr(self, 'debug_printed'):
             with torch.no_grad():
                 logits_test = self.classifier(cls_output)
                 print(f"\n[DEBUG Model] Logits mean: {logits_test.mean().item():.3f}, std: {logits_test.std().item():.3f}")
             self.debug_printed = True

        # Logs
        logits = self.classifier(cls_output)
        
        # Loss
        loss = nn.functional.cross_entropy(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }


def parse_depths(depths):
    if not depths or depths == "all":
        return ["all"]
    return [d.strip() for d in depths.split(",") if d.strip()]


def build_arg_parser(defaults=None, prog="python -m ruletaker.cli.train", description="Train on RuleTaker only."):
    defaults = defaults or RULETAKER_TRAIN_DEFAULTS
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path. CLI args override YAML values.")
    parser.add_argument("--model_type", type=str, default=defaults["model_type"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--root", type=str, default=defaults["root"], help="RuleTaker data root.")
    parser.add_argument("--gpus", type=str, default=defaults["gpus"])
    parser.add_argument("--limit", type=int, default=defaults["limit"])
    parser.add_argument(
        "--train_depth",
        type=str,
        default=defaults["train_depth"],
        help="e.g., depth-1,depth-2 or all",
    )
    parser.add_argument(
        "--eval_depths",
        type=str,
        default=defaults["eval_depths"],
        help="e.g., depth-0,depth-1,depth-2,depth-3,depth-3ext,depth-5 or all",
    )
    parser.add_argument(
        "--eval_on_train",
        action=argparse.BooleanOptionalAction,
        default=defaults["eval_on_train"],
        help="Use training split for evaluation (sanity check/overfit)",
    )
    parser.add_argument("--max_length", type=int, default=defaults["max_length"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    return parser


# ==========================================
# 3. Training Loop (Adapted)
# ==========================================
def parse_training_args(prog="python -m ruletaker.cli.train", description="Train on RuleTaker only."):
    config_path = _extract_config_path()
    defaults = dict(RULETAKER_TRAIN_DEFAULTS)

    yaml_config = _load_yaml_config(config_path)
    unknown_keys = sorted(set(yaml_config.keys()) - set(defaults.keys()))
    if unknown_keys:
        raise ValueError(f"Unknown config keys in {config_path}: {', '.join(unknown_keys)}")
    defaults.update(yaml_config)

    parser = build_arg_parser(defaults=defaults, prog=prog, description=description)
    return parser.parse_args()


def run_training(prog="python -m ruletaker.cli.train", description="Train on RuleTaker only."):
    args = parse_training_args(prog=prog, description=description)

    # GPU Setup
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = build_ruletaker_tokenizer(args.model_type)
    
    # Dataset
    print(f"Loading RuleTaker from: {args.root}")
    # NOTE: RuleTaker has 5 datasets, using "small" by default if not specified in load_dataset(path, name)
    # But user said load_dataset("tasksource/ruletaker"). We'll try default.
    collator = RuleTakerCollator(tokenizer, device, max_length=args.max_length)

    train_ds = RuleTakerDataset(split="train", limit=args.limit, depth=args.train_depth, dataset_dir=args.root)
    test_split = "train" if args.eval_on_train else "test"

    eval_depths = parse_depths(args.eval_depths)
    eval_loaders = {}
    for d in eval_depths:
        depth_arg = "all" if d == "all" else d
        eval_ds = RuleTakerDataset(split=test_split, limit=args.limit, depth=depth_arg, dataset_dir=args.root)
        eval_loaders[d] = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    # train_loader already uses train_depth; eval loaders are per-depth
    
    # Model
    model = RuleTakerClassifier(device, tokenizer, model_type=args.model_type, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = _make_train_pbar(train_loader, desc=f"Ep {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Acc
            preds = torch.argmax(out['logits'], dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += len(batch['labels'])
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.3f}"})
            
        # Eval
        print(f"Evaluating Epoch {epoch+1}...")
        model.eval()
        with torch.no_grad():
            for depth_name, loader in eval_loaders.items():
                eval_correct = 0
                eval_total = 0
                for batch in loader:
                    out = model(batch)
                    preds = torch.argmax(out['logits'], dim=1)
                    eval_correct += (preds == batch['labels']).sum().item()
                    eval_total += len(batch['labels'])

                acc = (eval_correct / eval_total) if eval_total > 0 else 0.0
                print(f"Dev/Test Acc [{depth_name}]: {acc:.4f}")

if __name__ == "__main__":
    run_training()
