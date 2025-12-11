import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import transformers

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from clutrr.data.dataset import CLUTRRDataset, relation_id_map
from clutrr.models.model_scratch import CLUTRRTransformerModel

TYPE_ENT = 1  # 必须和 graph_parsing.py 中的定义一致


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.num_relations = len(relation_id_map)

    print(f"Using device: {device}")
    print(f"Attention Type: {args.attention_type}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(script_dir, "../data/clutrr")

    train_dataset = CLUTRRDataset(root, args.dataset, "train", args.data_percentage)
    test_dataset = CLUTRRDataset(root, args.dataset, "test", 100)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=CLUTRRDataset.collate_fn,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=CLUTRRDataset.collate_fn,
        shuffle=False,
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    model = CLUTRRTransformerModel(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for (input_ids, attention_mask, type_labels, unify_mat, sub_indices, obj_indices), answers in pbar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            type_labels = type_labels.to(device)
            unify_mat = unify_mat.to(device)
            answers = answers.to(device)

            optimizer.zero_grad()

            task_logits, all_type_logits, all_unify_logits, _ = model(
                input_ids, attention_mask, sub_indices, obj_indices
            )

            # 1) 主任务损失
            task_loss = criterion(task_logits, answers)

            total_type_loss = torch.tensor(0.0, device=device)
            total_unify_loss = torch.tensor(0.0, device=device)

            if args.attention_type == "softunif":
                # 2) Type Loss：只在前 N 层监督
                if len(all_type_logits) > 0 and args.supervised_layers > 0:
                    sup_layers = min(args.supervised_layers, len(all_type_logits))
                    for i in range(sup_layers):
                        layer_logits = all_type_logits[i]  # [B,L,5]
                        total_type_loss += nn.CrossEntropyLoss()(
                            layer_logits.view(-1, layer_logits.size(-1)),
                            type_labels.view(-1),
                        )

                # 3) Unification Loss：行归一的交叉熵，只在实体行 + 至少两次出现上监督
                if len(all_unify_logits) > 0 and args.supervised_layers > 0:
                    B, L, _ = unify_mat.size()
                    row_valid = (attention_mask > 0)          # [B,L]
                    ent_mask = (type_labels == TYPE_ENT)      # [B,L]

                    for i, layer_logits in enumerate(all_unify_logits):
                        if i >= args.supervised_layers:
                            break

                        # layer_logits: [B,H,L,L] -> [B,L,L]
                        logits = layer_logits.mean(dim=1)

                        # mask 掉 padding 行列
                        row_mask = attention_mask.unsqueeze(2)   # [B,L,1]
                        col_mask = attention_mask.unsqueeze(1)   # [B,1,L]
                        logits_mask = row_mask * col_mask        # [B,L,L]

                        logits = logits.masked_fill(logits_mask == 0, -1e9)

                        target = unify_mat * logits_mask         # [B,L,L]
                        row_sum = target.sum(dim=-1, keepdim=True)  # [B,L,1]

                        target = torch.where(
                            row_sum > 0, target / (row_sum + 1e-9), target
                        )

                        log_probs = F.log_softmax(logits, dim=-1)   # [B,L,L]
                        ce_row = -(target * log_probs).sum(dim=-1)  # [B,L]

                        # 至少两次出现的实体行：row_sum > 1.5（因为对角线本身是 1）
                        multi_mask = (row_sum.squeeze(-1) > 1.5)        # [B,L]
                        valid_row = row_valid & ent_mask & multi_mask   # [B,L]

                        ce_row = ce_row * valid_row.float()
                        denom = valid_row.float().sum().clamp_min(1.0)
                        layer_unify_loss = ce_row.sum() / denom
                        total_unify_loss += layer_unify_loss

            loss = task_loss \
                + args.lambda_type * total_type_loss \
                + args.lambda_unify * total_unify_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(task_logits, dim=1)
            correct += (preds == answers).sum().item()
            total += answers.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss / max(total / args.batch_size, 1):.3f}",
                "acc": f"{correct / max(total, 1):.3f}",
                "task": f"{task_loss.item():.2f}",
                "type": f"{total_type_loss.item():.2f}",
                "unif": f"{total_unify_loss.item():.2f}",
            })

        train_acc = correct / total
        print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}  Train Acc: {train_acc:.4f}")

        # 验证
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for (input_ids, attention_mask, _, _, sub_indices, obj_indices), answers in test_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                answers = answers.to(device)

                task_logits, _, _, _ = model(input_ids, attention_mask, sub_indices, obj_indices)
                preds = torch.argmax(task_logits, dim=1)
                test_correct += (preds == answers).sum().item()
                test_total += answers.size(0)

        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1} Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            save_dir = os.path.join(script_dir, "./model/clutrr")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir, f"clutrr_transformer_{args.attention_type}_best.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_db9b8f04")
    parser.add_argument("--data_percentage", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # Model Config
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mlp_layers", type=int, default=1)

    parser.add_argument("--attention_type", type=str, default="softunif", choices=["vanilla", "softunif"])

    parser.add_argument("--lambda_type", type=float, default=0.5)
    parser.add_argument("--lambda_unify", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    # 只在前 N 层上施加 type / unify 监督
    parser.add_argument("--supervised_layers", type=int, default=2)

    args = parser.parse_args()

    # Seeding
    if args.seed is not None:
      torch.manual_seed(args.seed)
      torch.cuda.manual_seed_all(args.seed)
      random.seed(args.seed)
      np.random.seed(args.seed)
      transformers.set_seed(args.seed)

    train(args)
