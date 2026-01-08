
import pandas as pd
import ast
import re

MALE_ROLES = {
    'father', 'brother', 'son', 'husband', 'grandfather', 'grandson', 
    'uncle', 'nephew', 'brother-in-law', 'father-in-law', 'son-in-law'
}
FEMALE_ROLES = {
    'mother', 'sister', 'daughter', 'wife', 'grandmother', 'granddaughter', 
    'aunt', 'niece', 'sister-in-law', 'mother-in-law', 'daughter-in-law'
}

def get_names_in_appearance_order(story):
    # Extract names in brackets like [Name]
    seen = set()
    ordered = []
    for m in re.finditer(r"\[(.*?)\]", story):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered

def get_names_from_genders(genders_str):
    if pd.isna(genders_str): return []
    # format: "Name:gender,Name:gender"
    return [item.split(':')[0] for item in genders_str.split(',')]

def get_gender_map(genders_str):
    if pd.isna(genders_str): return {}
    g_map = {}
    for item in genders_str.split(','):
        if ':' in item:
            name, gender = item.split(':')
            g_map[name] = gender
    return g_map

def check_consistency(names, edges, edge_types, gender_map):
    violations = 0
    details = []
    
    for i, (u, v) in enumerate(edges):
        if i >= len(edge_types): break
        rel = edge_types[i]
        
        try:
            name_u = names[u]
            name_v = names[v]
        except IndexError:
            violations += 1
            details.append(f"IndexError for edge {i}: ({u},{v}) with len(names)={len(names)}")
            continue

        gen_u = gender_map.get(name_u, 'unknown')
        gen_v = gender_map.get(name_v, 'unknown')
        
        # Check strict gender constraints
        # 1. Same-sex requirements
        if rel in ['husband', 'wife']:
            if gen_u == gen_v and gen_u != 'unknown':
                violations += 1
                details.append(f"Edge ({name_u}, {name_v}) type '{rel}' but genders are ({gen_u}, {gen_v})")
                continue
                
        # 2. Role constraints
        # At least one person must satisfy the gender of the role.
        # e.g. for 'sister', either u or v must be female.
        # If both are male, impossible.
        if rel in FEMALE_ROLES:
            if gen_u == 'male' and gen_v == 'male':
                violations += 1
                details.append(f"Edge ({name_u}, {name_v}) type '{rel}' but both are male")
                continue
                
        if rel in MALE_ROLES:
            if gen_u == 'female' and gen_v == 'female':
                violations += 1
                details.append(f"Edge ({name_u}, {name_v}) type '{rel}' but both are female")
                continue

    return violations, details

def main():
    csv_path = 'data/clutrr/data_089907f8/1.2,1.3_train.csv'
    df = pd.read_csv(csv_path)
    
    hyp_a_failures = 0
    hyp_b_failures = 0
    total_rows = 20
    
    print(f"{'Row':<4} | {'Valid A (Appear)':<16} | {'Valid B (Genders)':<16} | {'Details'}")
    print("-" * 80)
    
    for i in range(total_rows):
        row = df.iloc[i]
        story = row['story']
        story_edges = ast.literal_eval(row['story_edges'])
        edge_types = ast.literal_eval(row['edge_types'])
        genders_str = row['genders']
        
        names_a = get_names_in_appearance_order(story)
        names_b = get_names_from_genders(genders_str)
        gender_map = get_gender_map(genders_str)
        
        viol_a, det_a = check_consistency(names_a, story_edges, edge_types, gender_map)
        viol_b, det_b = check_consistency(names_b, story_edges, edge_types, gender_map)
        
        is_a_valid = (viol_a == 0) and (names_a == names_b if len(names_a)==len(names_b) else True) # Note: Names lists might differ in order
        # Wait, if names lists are same order, both are valid. If different, one might fail.
        
        # Stricter: Just use violations.
        status_a = "OK" if viol_a == 0 else "FAIL"
        status_b = "OK" if viol_b == 0 else "FAIL"
        
        extra_info = ""
        if viol_a > 0:
            hyp_a_failures += 1
            extra_info += f"A: {det_a[0]} "
        if viol_b > 0:
            hyp_b_failures += 1
            extra_info += f"B: {det_b[0]} "
            
        # Check if orders are different
        note = ""
        if names_a != names_b:
            note = "[Order Differs]"
        
        print(f"{i:<4} | {status_a:<16} | {status_b:<16} | {note} {extra_info[:50]}...")

    print("-" * 80)
    print(f"Total Rows: {total_rows}")
    print(f"Hypothesis A (Appearance Order) Failures: {hyp_a_failures}")
    print(f"Hypothesis B (Genders List Order) Failures: {hyp_b_failures}")

if __name__ == "__main__":
    main()
