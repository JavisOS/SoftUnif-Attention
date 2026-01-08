import os
import csv
import ast
import networkx as nx
from collections import Counter

target_dir = "/home/batchcom/dsr-lm/data/clutrr/data_089907f8/"

if not os.path.exists(target_dir):
    print(f"Directory not found: {target_dir}")
    exit(1)

files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
files.sort()

print(f"{'File':<30} | {'Total':<6} | {'Hops Distribution'}")
print("-" * 120)

for filename in files:
    filepath = os.path.join(target_dir, filename)
    hops_counter = Counter()
    total_rows = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            continue
            
        # Determine indices dynamically if possible, else fallback
        try:
            edge_idx = header.index('story_edges')
            q_edge_idx = header.index('query_edge')
        except ValueError:
            # Fallback based on visual inspection
            edge_idx = 11
            q_edge_idx = 13
        
        for row in reader:
            if len(row) <= max(edge_idx, q_edge_idx):
                continue
                
            try:
                story_edges_str = row[edge_idx]
                query_edge_str = row[q_edge_idx]
                
                if not story_edges_str or not query_edge_str:
                    continue

                edges = ast.literal_eval(story_edges_str)
                q_edge = ast.literal_eval(query_edge_str)
                
                # Check format
                if not isinstance(q_edge, tuple) or len(q_edge) != 2:
                    continue
                
                start_node, end_node = q_edge
                
                G = nx.Graph()
                G.add_edges_from(edges)
                
                if start_node not in G: G.add_node(start_node)
                if end_node not in G: G.add_node(end_node)
                
                try:
                    path_len = nx.shortest_path_length(G, source=start_node, target=end_node)
                    hops_counter[path_len] += 1
                except nx.NetworkXNoPath:
                    hops_counter['inf'] += 1
                
                total_rows += 1
            except Exception as e:
                # print(f"Error parsing row in {filename}: {e}")
                continue

    # Format distribution
    # Sort by hop count
    sorted_items = sorted(hops_counter.items(), key=lambda x: (x[0] if isinstance(x[0], int) else 999))
    dist_str = ", ".join([f"{k}: {v}" for k, v in sorted_items])
    print(f"{filename:<30} | {total_rows:<6} | {dist_str}")
