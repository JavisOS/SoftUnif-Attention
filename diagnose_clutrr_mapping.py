
import pandas as pd
import ast
import re

def get_names_in_order(story):
    # Extract names in brackets like [Name]
    return [m.group(1) for m in re.finditer(r"\[(.*?)\]", story)]

def analyze_row(row_idx, row):
    print(f"\n--- Row {row_idx} ---")
    story = row['story']
    story_edges = ast.literal_eval(row['story_edges'])
    edge_types = ast.literal_eval(row['edge_types'])
    node_mapping = ast.literal_eval(row['node_mapping']) if isinstance(row['node_mapping'], str) else row['node_mapping']
    genders_str = row['genders']
    
    print(f"Story: {story}")
    print(f"Edges: {story_edges}")
    print(f"Types: {edge_types}")
    print(f"Node Mapping: {node_mapping}")
    
    # 1. Appearance Order (Unique)
    mentions = get_names_in_order(story)
    unique_names_appearance = []
    for name in mentions:
        if name not in unique_names_appearance:
            unique_names_appearance.append(name)
            
    print(f"Unique Names (Appearance): {unique_names_appearance}")
    
    # Check Hypothesis A: ID corresponds to index in unique_names_appearance
    print("Testing Hypothesis A (ID = Appearance Index):")
    for i, (u, v) in enumerate(story_edges):
        if i >= len(edge_types): break
        rel_type = edge_types[i]
        
        try:
            name_u = unique_names_appearance[u]
            name_v = unique_names_appearance[v]
            print(f"  Edge {i}: ({u},{v}) -> {name_u} is '{rel_type}' of {name_v}?")
        except IndexError:
            print(f"  Edge {i}: ({u},{v}) -> Index out of bounds for names list.")

    # Check Hypothesis B: IDs map to names via gender list order?
    # genders format: "Name:gender,Name:gender"
    if pd.isna(genders_str):
        print("Genders is NaN")
        gender_names = []
    else:
        gender_items = genders_str.split(',')
        gender_names = [item.split(':')[0] for item in gender_items]
    
    print(f"Gender Names List: {gender_names}")
    print("Testing Hypothesis B (ID = Gender List Index):")
    for i, (u, v) in enumerate(story_edges):
        if i >= len(edge_types): break
        rel_type = edge_types[i]
        
        try:
            name_u = gender_names[u]
            name_v = gender_names[v]
            print(f"  Edge {i}: ({u},{v}) -> {name_u} is '{rel_type}' of {name_v}?")
        except IndexError:
            print(f"  Edge {i}: ({u},{v}) -> Index out of bounds for gender names list.")

def main():
    csv_path = 'data/clutrr/data_089907f8/1.2,1.3_train.csv'
    df = pd.read_csv(csv_path)
    
    for i in range(20):
        analyze_row(i, df.iloc[i])

if __name__ == "__main__":
    main()
