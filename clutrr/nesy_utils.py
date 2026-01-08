
import networkx as nx
import numpy as np
import ast
import re

def get_names_from_genders(genders_str):
    """
    Parses 'Name:gender,Name2:gender...' string into list of names.
    This list DEFINES the index mapping for story_edges.
    """
    if not isinstance(genders_str, str): return []
    # Items are comma separated
    # Some items might not have colon if malformed?
    names = []
    items = genders_str.split(',')
    for item in items:
        # Expected 'Name:gender'
        # Split on LAST colon in case name has colon? Unlikely.
        if ':' in item:
            name = item.split(':')[0].strip()
            names.append(name)
        else:
            names.append(item.strip()) # Fallback
    return names

def parse_graph_and_path(row):
    """
    Parses a single CSV row to extract the ground truth reasoning path.
    
    Args:
        row: row list/dict from CLUTRR CSV. 
             If list (from csv.reader): indices specific to current columns.
             If dict (from pandas or mapped): keys.
             
             Assuming list access based on baseline_roberta_analysis:
             Reference:
             row[2] = story
             row[3] = query ('S', 'O')
             row[11] = story_edges "[(0,1)...]"
             row[12] = edge_types "['rel' ...]"
             row[14] = genders_str
             
    Returns:
        dict with:
            'path_node_indices': list of int (indices in all_names)
            'path_relation_labels': list of str
            'all_names': list of str (ordered by index)
            'query_indices': (start_idx, end_idx)
    """
    # Defensive parsing
    try:
        story = row[2]
        query_str = row[3] # "('Sub', 'Obj')"
        edges_str = row[11]
        types_str = row[12]
        genders_str = row[14]
        
        # 1. Parse Names Mapping
        all_names = get_names_from_genders(genders_str)
        if not all_names: 
            return None
        
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        
        # 2. Parse Query
        # query_str is literal tuple string
        try:
            sub, obj = ast.literal_eval(query_str)
        except:
            # Fallback if already tuple
            sub, obj = query_str
            
        if sub not in name_to_idx or obj not in name_to_idx:
            return None
            
        start_node = name_to_idx[sub]
        end_node = name_to_idx[obj]
        
        # 3. Build Graph
        G = nx.Graph() # Undirected as kinship is symmetric in terms of existence (but directed relations)
        # Actually CLUTRR edges are directed 0->1 "father" means 0 is father of 1.
        # But for Shortest path finding, we treat as undirected to find connectivity, 
        # then check directions? 
        # Ideally, Reasoning is A -> B -> C.
        # Edges might be stored as (C, B) 'son' which is B -> C 'father'.
        # Scallop logic handles inverses. Here we just want the chain of entities.
        # Shortest path on Undirected graph gives the chain of entities.
        
        raw_edges = ast.literal_eval(edges_str)
        raw_types = ast.literal_eval(types_str)
        
        for (u, v), t in zip(raw_edges, raw_types):
            G.add_edge(u, v, type=t)
            
        # 4. Find Path
        try:
            path_indices = nx.shortest_path(G, source=start_node, target=end_node)
        except nx.NetworkXNoPath:
            return None
            
        # 5. Extract Relations for Path (Optional, for Edge Loss)
        # For each hop u->v, find valid edge type
        path_relations = []
        for i in range(len(path_indices) - 1):
            u, v = path_indices[i], path_indices[i+1]
            data = G.get_edge_data(u, v)
            if data:
                path_relations.append(data['type'])
            else:
                path_relations.append("unknown")
                
        return {
            'path_node_indices': path_indices,
            'path_relation_labels': path_relations,
            'all_names': all_names
        }
    except Exception as e:
        # print(f"Parse error: {e}")
        return None

def apply_bijective_map(text, mapping):
    """
    Applies bijective renaming to text using mapping dict.
    Safe replacement using regex to avoid partial overlaps.
    """
    # Sort keys by length descending
    sorted_names = sorted(mapping.keys(), key=len, reverse=True)
    escaped_names = [re.escape(n) for n in sorted_names]
    
    if not escaped_names: return text
    
    pattern = re.compile(r'\b(' + '|'.join(escaped_names) + r')\b')
    
    def replacement(match):
        return mapping[match.group(0)]
        
    return pattern.sub(replacement, text)

import random

def augment_bijective_swap(story, query, all_names):
    """
    Creates a counterfactual version of story and query by cycling names.
    Returns: (new_story, new_query_tuple)
    """
    if len(all_names) < 2:
        return story, query
        
    # Rotate names: A->B, B->C ... Z->A
    # Or shuffle? Cyclical shift guarantees everyone changes (if len > 1) which is good.
    # While random shuffle might map A->A.
    # We prefer "Change" to "Random".
    
    # Clone and Shuffle
    # To ensure robust change, we can perform a random Derangement or just Cycle.
    # Simple Cycle is efficient and sufficient.
    
    shift = 1 # Shift by 1
    shuffled = all_names[shift:] + all_names[:shift]
    
    mapping = {n: s for n, s in zip(all_names, shuffled)}
    
    new_story = apply_bijective_map(story, mapping)
    
    q_sub, q_obj = query
    new_q_sub = mapping.get(q_sub, q_sub)
    new_q_obj = mapping.get(q_obj, q_obj)
    
    return new_story, (new_q_sub, new_q_obj)

