import os
import csv

from argparse import ArgumentParser
from clutrr.utils.parsing import safe_literal_eval

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--root", type=str, default="data")
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  args = parser.parse_args()

  # Load files
  dataset_dir = os.path.abspath(os.path.join(args.root, args.dataset))
  file_names = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if ".csv" in d]

  # Load proof states
  all_relations = {}
  for file_name in file_names:
    file_csv_content = csv.reader(open(file_name))
    for row in list(file_csv_content)[1:]:
      relations = safe_literal_eval(row[12], default=[])
      target_relation = row[5]
      for r in relations + [target_relation]:
        if r in all_relations:
          all_relations[r] += 1
        else:
          all_relations[r] = 1

  # Print the result
  print(f"Number of relations: {len(all_relations)}")
  print("Relation name and its count:", all_relations)
