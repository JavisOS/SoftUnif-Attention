import networkx as nx
import numpy as np
import ast
import re
import random

# Mapping for Inverses: Relation -> {SourceGender -> InverseRelation}
# "SourceGender" refers to the gender of the start_node of the INVERSE edge.
# i.e. if Original is u->v (type T), Inverse is v->u (type InvT).
# We check gender of v to determine InvT (inverse edge source).
RELATION_INVERSES = {
    'father': {'male': 'son', 'female': 'daughter'},
    'mother': {'male': 'son', 'female': 'daughter'},
    'son': {'male': 'father', 'female': 'mother'},
    'daughter': {'male': 'father', 'female': 'mother'},
    'brother': {'male': 'brother', 'female': 'sister'},
    'sister': {'male': 'brother', 'female': 'sister'},
    'grandfather': {'male': 'grandson', 'female': 'granddaughter'},
    'grandmother': {'male': 'grandson', 'female': 'granddaughter'},
    'grandson': {'male': 'grandfather', 'female': 'grandmother'},
    'granddaughter': {'male': 'grandfather', 'female': 'grandmother'},
    'husband': {'male': 'wife', 'female': 'wife'},
    'wife': {'male': 'husband', 'female': 'husband'},
    'uncle': {'male': 'nephew', 'female': 'niece'},
    'aunt': {'male': 'nephew', 'female': 'niece'},
    'nephew': {'male': 'uncle', 'female': 'aunt'},
    'niece': {'male': 'uncle', 'female': 'aunt'},
    'son-in-law': {'male': 'father-in-law', 'female': 'mother-in-law'},
    'daughter-in-law': {'male': 'father-in-law', 'female': 'mother-in-law'},
    'father-in-law': {'male': 'son-in-law', 'female': 'daughter-in-law'},
    'mother-in-law': {'male': 'son-in-law', 'female': 'daughter-in-law'},
}

def get_name_gender_map(genders_str):
    """
    Parses 'Name:gender,Name2:gender...' string.
    Returns:
        names_ordered: list of names (defines indices)
        name_gender: dict {name: gender}
    """
    if not isinstance(genders_str, str): return [], {}
    names_ordered = []
    name_gender = {}
    
    items = genders_str.split(',')
    for item in items:
        # Split on last colon
        last_colon = item.rfind(':')
        if last_colon != -1:
            name = item[:last_colon].strip()
            gender = item[last_colon+1:].strip().lower()
            names_ordered.append(name)
            name_gender[name] = gender
        else:
            # Fallback
            name = item.strip()
            names_ordered.append(name)
            name_gender[name] = 'unknown'
            
    return names_ordered, name_gender

def get_names_from_genders(genders_str):
    # Compatibility wrapper
    names, _ = get_name_gender_map(genders_str)
    return names

def parse_graph_and_path(row):
    """
    Parses a single CSV row to extract the ground truth reasoning path.
    Handles BI-DIRECTIONAL edges by inferring inverse relations using gender.
    """
    try:
        # row indices based on baseline_roberta_analysis / csv format
        story = row[2]
        query_str = row[3]
        edges_str = row[11]
        types_str = row[12]
        genders_str = row[14]
        
        # 1. Parse Names & Genders
        all_names, name_gender = get_name_gender_map(genders_str)
        if not all_names: 
            return None
        
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        
        # 2. Parse Query
        try:
            sub, obj = ast.literal_eval(query_str)
        except:
            sub, obj = query_str
            
        if sub not in name_to_idx or obj not in name_to_idx:
            return None
            
        start_node = name_to_idx[sub]
        end_node = name_to_idx[obj]
        
        # 3. Build Directed Graph with Bidirectional Edges
        # Need DiGraph to distinguish u->v (type) from v->u (inverse type)
        G = nx.DiGraph()
        
        # raw_edges: [(0,1), (1,2)...]
        raw_edges = ast.literal_eval(edges_str)
        raw_types = ast.literal_eval(types_str)
        
        for (u, v), t in zip(raw_edges, raw_types):
            t_lower = t.lower()
            # Forward: u -> v is t
            G.add_edge(u, v, type=t_lower)
            
            # Backward: v -> u is inverse(t)
            # v is source of backward edge. Need gender of v.
            if 0 <= v < len(all_names):
                v_name = all_names[v]
                v_gender = name_gender.get(v_name, 'unknown')
                
                inv_t = f"inverse_{t_lower}" # default
                if t_lower in RELATION_INVERSES:
                    inv_map = RELATION_INVERSES[t_lower]
                    if v_gender in inv_map:
                        inv_t = inv_map[v_gender]
                
                # Check if edge already exists? Usually not in kinship trees unless cycle.
                # If cycle exists (e.g. husband/wife), both might be defined or just one.
                # CLUTRR usually defines one.
                G.add_edge(v, u, type=inv_t)
            
        # 4. Find Shortest Path
        try:
            # Use shortest path on DiGraph
            path_indices = nx.shortest_path(G, source=start_node, target=end_node)
        except nx.NetworkXNoPath:
            # Fallback: Try undirected just for connectivity?
            # If undirected works but directed doesn't, it means we are missing inverse logic.
            # But we added inverses for ALL edges. So connectivity should be same as Undirected.
            return None
            
        # 5. Extract Relations
        path_relations = []
        for i in range(len(path_indices) - 1):
            u_node, v_node = path_indices[i], path_indices[i+1]
            data = G.get_edge_data(u_node, v_node)
            rel = data['type'] if data else "unknown"
            path_relations.append(rel)
                
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
    Safe replacement using regex.
    """
    sorted_names = sorted(mapping.keys(), key=len, reverse=True)
    if not sorted_names: return text
    
    # Escape for regex
    escaped_names = [re.escape(n) for n in sorted_names]
    pattern = re.compile(r'\b(' + '|'.join(escaped_names) + r')\b')
    
    def replacement(match):
        return mapping[match.group(0)]
        
    return pattern.sub(replacement, text)

def augment_bijective_swap(story, query, all_names):
    if len(all_names) < 2:
        return story, query
        
    # Cycle shift
    shift = 1 
    shuffled = all_names[shift:] + all_names[:shift]
    
    mapping = {n: s for n, s in zip(all_names, shuffled)}
    
    new_story = apply_bijective_map(story, mapping)
    
    q_sub, q_obj = query
    new_q_sub = mapping.get(q_sub, q_sub)
    new_q_obj = mapping.get(q_obj, q_obj)
    
    return new_story, (new_q_sub, new_q_obj)
