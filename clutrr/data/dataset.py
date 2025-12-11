import os
import csv
import math
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizerFast
from clutrr.data.graph_parsing import process_clutrr_row
from tqdm import tqdm
import multiprocessing

# Disable tokenizer parallelism to avoid deadlocks in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define relation map
relation_id_map = {
    'daughter': 0,
    'sister': 1,
    'son': 2,
    'aunt': 3,
    'father': 4,
    'husband': 5,
    'granddaughter': 6,
    'brother': 7,
    'nephew': 8,
    'mother': 9,
    'uncle': 10,
    'grandfather': 11,
    'wife': 12,
    'grandmother': 13,
    'niece': 14,
    'grandson': 15,
    'son-in-law': 16,
    'father-in-law': 17,
    'daughter-in-law': 18,
    'mother-in-law': 19,
    'nothing': 20,
}

# Global variable for worker processes
worker_tokenizer = None


def init_worker():
    global worker_tokenizer
    # Suppress tokenizer parallelism warning inside workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    worker_tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base", add_prefix_space=True
    )


def preprocess_item(row):
    """
    row:
        [id, story, query, text_query, target, text_target, ...]
        we use:
            row[2]: story with [...] entities
            row[3]: query tuple string, e.g. "('Donald', 'Dorothy')"
            row[5]: target relation string
    """
    global worker_tokenizer

    # Query is of type (sub, obj)
    query_sub_obj = eval(row[3])

    # *** IMPORTANT ***
    # Do NOT .lower() here; entity_to_tokens keys come from bracketed names,
    # e.g. "[Donald]" -> "Donald", so we must keep the exact form.
    sub_name = str(query_sub_obj[0])
    obj_name = str(query_sub_obj[1])
    query = (sub_name, obj_name)

    # Answer is one of 20 classes
    answer = row[5]

    # Graph Parsing for Supervision
    processed = process_clutrr_row(row[2], worker_tokenizer)
    input_ids = processed['input_ids'].squeeze(0)       # [Seq]
    attention_mask = processed['attention_mask'].squeeze(0)  # [Seq]
    type_labels = processed['type_labels']              # [Seq]
    unification_matrix = processed['unification_matrix']  # [Seq, Seq]
    entity_to_tokens = processed['entity_to_tokens']    # dict name -> [indices]

    # Get token indices for query subject and object
    sub_indices = entity_to_tokens.get(sub_name, [])
    obj_indices = entity_to_tokens.get(obj_name, [])

    return (
        (input_ids, attention_mask, type_labels,
         unification_matrix, sub_indices, obj_indices),
        answer
    )


class CLUTRRDataset(Dataset):
    def __init__(self, root, dataset, split, data_percentage, tokenizer=None):
        self.dataset_dir = os.path.join(root, f"{dataset}/")
        # Filter files based on split
        self.file_names = [
            os.path.join(self.dataset_dir, d)
            for d in os.listdir(self.dataset_dir)
            if f"_{split}.csv" in d
        ]
        self.data = [
            row
            for f in self.file_names
            for row in list(csv.reader(open(f)))[1:]
        ]
        self.data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[:self.data_num]

        if tokenizer is None:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-base", add_prefix_space=True
            )
        else:
            self.tokenizer = tokenizer

        # Cache check
        cache_dir = os.path.join(root, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir, f"cached_{dataset}_{split}_{data_percentage}.pt"
        )

        if os.path.exists(cache_file):
            print(f"Loading cached {split} dataset from {cache_file}.")
            self.processed_data = torch.load(cache_file)
        else:
            # Pre-process data 
            self.processed_data = []
            num_workers = min(multiprocessing.cpu_count(), 8)  # cap at 8
            print(
                f"Pre-processing {dataset} {split} dataset with {num_workers} workers."
            )

            with multiprocessing.Pool(
                processes=num_workers, initializer=init_worker
            ) as pool:
                self.processed_data = list(
                    tqdm(pool.imap(preprocess_item, self.data),
                         total=len(self.data))
                )

            print(f"Saving cached {split} dataset to {cache_file}.")
            torch.save(self.processed_data, cache_file)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, i):
        return self.processed_data[i]

    @staticmethod
    def collate_fn(batch):
        answers = torch.stack(
            [torch.tensor(relation_id_map[answer]) for (_, answer) in batch]
        )

        # Batch Input IDs
        input_ids_list = [item[0][0] for item in batch]
        input_ids_batch = pad_sequence(
            input_ids_list, batch_first=True, padding_value=1
        )  # RoBERTa pad_token_id = 1

        # Batch Attention Mask
        mask_list = [item[0][1] for item in batch]
        mask_batch = pad_sequence(
            mask_list, batch_first=True, padding_value=0
        )

        # Batch Type Labels
        type_labels_list = [item[0][2] for item in batch]
        type_labels_batch = pad_sequence(
            type_labels_list, batch_first=True, padding_value=0
        )

        # Batch Unification Matrices
        unify_mats_list = [item[0][3] for item in batch]
        max_len = max([m.size(0) for m in unify_mats_list])
        unify_mat_batch = torch.zeros(len(batch), max_len, max_len)
        for i, m in enumerate(unify_mats_list):
            l = m.size(0)
            unify_mat_batch[i, :l, :l] = m

        # Batch Query Indices (list of lists)
        sub_indices = [item[0][4] for item in batch]
        obj_indices = [item[0][5] for item in batch]

        return (
            (input_ids_batch, mask_batch, type_labels_batch,
             unify_mat_batch, sub_indices, obj_indices),
            answers,
        )
