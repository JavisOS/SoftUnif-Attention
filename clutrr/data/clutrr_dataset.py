"""Shared CLUTRR dataset utilities used by TRUA and comparison baselines."""

import csv
import math
import os

import torch
from torch.utils.data import DataLoader, Dataset

from clutrr.utils.parsing import parse_pair_literal, safe_literal_eval
from clutrr.config.relation_schema import RELATION_ID_MAP_21_WITH_NOTHING as relation_id_map


def official_hop_count(row):
    """Return the CLUTRR task length recorded by the released dataset."""
    try:
        task_name = str(row[10])
        return int(task_name.rsplit(".", 1)[-1])
    except (IndexError, TypeError, ValueError):
        edges = safe_literal_eval(row[11], default=None)
        if edges is None:
            raise ValueError("CLUTRR row has neither a valid task name nor story edges")
        return len(edges)


class CLUTRRDataset(Dataset):
    def __init__(self, root, dataset, split, data_percentage):
        self.dataset_dir = os.path.join(root, f"{dataset}/")
        if os.path.exists(self.dataset_dir):
            self.file_names = sorted(
                os.path.join(self.dataset_dir, d)
                for d in os.listdir(self.dataset_dir)
                if f"_{split}.csv" in d
            )
            self.data = []
            for file_name in self.file_names:
                with open(file_name, "r", encoding="utf-8") as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader, None)
                    self.data.extend(reader)
        else:
            self.file_names = []
            self.data = []

        self.data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[:self.data_num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        context_str = self.data[i][2]
        context = [s.strip().lower() for s in context_str.split(".") if s.strip() != ""]

        query_sub_obj = parse_pair_literal(self.data[i][3])
        if query_sub_obj is None:
            raise ValueError(f"Invalid query tuple format: {self.data[i][3]}")
        query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

        answer = self.data[i][5]

        hops = official_hop_count(self.data[i])

        return ((context, query), answer, hops, context_str)

    @staticmethod
    def collate_fn(batch):
        queries = [query for ((_, query), _, _, _) in batch]
        context_strs = [ctx_str for ((_, _), _, _, ctx_str) in batch]
        contexts = [fact for ((context, _), _, _, _) in batch for fact in context]
        context_lens = [len(context) for ((context, _), _, _, _) in batch]
        context_splits = [(sum(context_lens[:i]), sum(context_lens[:i + 1])) for i in range(len(context_lens))]
        answers = torch.stack([torch.tensor(relation_id_map[answer]) for (_, answer, _, _) in batch])
        hops = torch.tensor([h for (_, _, h, _) in batch])
        return ((contexts, queries, context_splits, context_strs), answers, hops)


def clutrr_loader(root, dataset, batch_size, training_data_percentage):
    train_dataset = CLUTRRDataset(root, dataset, "train", training_data_percentage)
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        collate_fn=CLUTRRDataset.collate_fn,
        shuffle=True,
        num_workers=0,
    )

    test_dataset = CLUTRRDataset(root, dataset, "test", 100)
    test_loader = DataLoader(
        test_dataset,
        batch_size,
        collate_fn=CLUTRRDataset.collate_fn,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader
