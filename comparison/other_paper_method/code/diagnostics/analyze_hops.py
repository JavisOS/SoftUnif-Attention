import argparse
import csv
import os
import sys
from collections import Counter

import networkx as nx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from clutrr.utils.parsing import safe_literal_eval


def analyze_directory(target_dir):
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    files = sorted(f for f in os.listdir(target_dir) if f.endswith(".csv"))

    print(f"{'File':<30} | {'Total':<6} | {'Hops Distribution'}")
    print("-" * 120)

    for filename in files:
        filepath = os.path.join(target_dir, filename)
        hops_counter = Counter()
        total_rows = 0

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue

            try:
                edge_idx = header.index("story_edges")
                q_edge_idx = header.index("query_edge")
            except ValueError:
                edge_idx = 11
                q_edge_idx = 13

            for row in reader:
                if len(row) <= max(edge_idx, q_edge_idx):
                    continue

                edges = safe_literal_eval(row[edge_idx], default=None)
                query_edge = safe_literal_eval(row[q_edge_idx], default=None)
                if edges is None or query_edge is None:
                    continue
                if not isinstance(query_edge, tuple) or len(query_edge) != 2:
                    continue

                start_node, end_node = query_edge
                graph = nx.Graph()
                graph.add_edges_from(edges)

                if start_node not in graph:
                    graph.add_node(start_node)
                if end_node not in graph:
                    graph.add_node(end_node)

                try:
                    path_len = nx.shortest_path_length(graph, source=start_node, target=end_node)
                    hops_counter[path_len] += 1
                except nx.NetworkXNoPath:
                    hops_counter["inf"] += 1
                total_rows += 1

        sorted_items = sorted(hops_counter.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 999))
        dist_str = ", ".join(f"{k}: {v}" for k, v in sorted_items)
        print(f"{filename:<30} | {total_rows:<6} | {dist_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, default="data/data_089907f8")
    args = parser.parse_args()
    analyze_directory(args.target_dir)


if __name__ == "__main__":
    main()
