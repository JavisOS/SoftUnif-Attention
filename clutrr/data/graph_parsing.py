import torch
import re
import logging

# Define Logic Ontology (Type Set)
# 0: NOISE
# 1: ENT
# 2: REL_VERT
# 3: REL_HOR
# 4: REL_MIX

TYPE_NOISE = 0
TYPE_ENT = 1
TYPE_REL_VERT = 2
TYPE_REL_HOR = 3
TYPE_REL_MIX = 4

# Relation Lists
REL_VERT = {
    'father', 'mother', 'son', 'daughter', 
    'grandfather', 'grandmother', 'grandson', 'granddaughter'
}
REL_HOR = {
    'brother', 'sister', 'husband', 'wife'
}
REL_MIX = {
    'uncle', 'aunt', 'nephew', 'niece',
    'father-in-law', 'mother-in-law', 'son-in-law', 'daughter-in-law'
}

# Combine all relations for searching, sorted by length descending to match "father-in-law" before "father"
ALL_RELATIONS = []
for r in REL_VERT: ALL_RELATIONS.append((r, TYPE_REL_VERT))
for r in REL_HOR: ALL_RELATIONS.append((r, TYPE_REL_HOR))
for r in REL_MIX: ALL_RELATIONS.append((r, TYPE_REL_MIX))

# Sort by length descending
ALL_RELATIONS.sort(key=lambda x: len(x[0]), reverse=True)

def get_clean_story_and_entities(raw_story):
    """
    Parses raw story with brackets like "[Alice] is [Bob]'s mother"
    Returns:
        clean_story: "Alice is Bob's mother"
        entity_map: dict {name: [(start, end), ...]} in clean_story coordinates
    """
    clean_story = ""
    entity_map = {}
    
    # Split by brackets to isolate entities
    # Example: "[Alice] is [Bob]'s mother" -> ['', '[Alice]', ' is ', '[Bob]', "'s mother"]
    parts = re.split(r"(\[.*?\])", raw_story)
    
    for part in parts:
        if not part:
            continue
            
        if part.startswith('[') and part.endswith(']'):
            # This is an entity
            name = part[1:-1] # Remove brackets
            start = len(clean_story)
            clean_story += name
            end = len(clean_story)
            
            if name not in entity_map:
                entity_map[name] = []
            entity_map[name].append((start, end))
        else:
            # This is normal text
            clean_story += part
            
    return clean_story, entity_map

def get_relation_spans(clean_story):
    """
    Finds relation words in clean story.
    Returns list of (start, end, type_id)
    """
    spans = []
    occupied_mask = [False] * len(clean_story)
    
    clean_story_lower = clean_story.lower()
    
    for rel_word, rel_type in ALL_RELATIONS:
        # Use word boundaries to avoid matching "son" inside "grandson"
        # However, "father-in-law" contains hyphens which are word boundaries.
        # We need to be careful. 
        # Simple regex with \b works for "father" vs "grandfather".
        # For "father-in-law", \bfather-in-law\b works.
        
        pattern = r"\b" + re.escape(rel_word) + r"\b"
        for match in re.finditer(pattern, clean_story_lower):
            start, end = match.span()
            
            # Check if overlap with already found longer relations
            is_occupied = any(occupied_mask[i] for i in range(start, end))
            if not is_occupied:
                spans.append((start, end, rel_type))
                # Mark occupied
                for i in range(start, end):
                    occupied_mask[i] = True
                    
    return spans

def process_clutrr_row(raw_story, tokenizer, max_len=512):
    """
    Process a single CLUTRR raw story string into model inputs and ground truth labels.
    
    Args:
        raw_story (str): String like "[Alice] is [Bob]'s mother."
        tokenizer: HuggingFace tokenizer (e.g. RobertaTokenizerFast)
        max_len (int): Max sequence length
        
    Returns:
        dict: {
            'input_ids': Tensor [1, Seq],
            'attention_mask': Tensor [1, Seq],
            'type_labels': Tensor [Seq],
            'unification_matrix': Tensor [Seq, Seq]
        }
    """
    # 1. Parse Raw Story
    clean_story, entity_map = get_clean_story_and_entities(raw_story)
    
    # 2. Find Relations
    relation_spans = get_relation_spans(clean_story)
    
    # 3. Tokenize
    # We use return_offsets_mapping to align tokens to character spans
    encoding = tokenizer(
        clean_story,
        padding=False,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'][0]
    attention_mask = encoding['attention_mask'][0]
    offsets = encoding['offset_mapping'][0] # [Seq, 2]
    
    seq_len = input_ids.size(0)
    
    # 4. Generate Labels
    type_labels = torch.zeros(seq_len, dtype=torch.long) # Default 0 (NOISE)
    
    # We also need to track which tokens belong to which entity for the unification matrix
    # token_to_entity: dict { token_idx: entity_name }
    token_to_entity = {}
    
    # Get tokens for pronoun checking
    tokens_list = tokenizer.convert_ids_to_tokens(input_ids)
    PRONOUNS = {'he', 'she', 'her', 'his', 'him'}
    
    for idx, (start, end) in enumerate(offsets):
        if start == end: # Special tokens (CLS, SEP, PAD)
            continue
            
        # Check Entity Overlap
        is_entity = False
        for name, spans in entity_map.items():
            for (e_start, e_end) in spans:
                # Check intersection
                # Token span: [start, end)
                # Entity span: [e_start, e_end)
                if max(start, e_start) < min(end, e_end):
                    type_labels[idx] = TYPE_ENT
                    token_to_entity[idx] = name
                    is_entity = True
                    break
            if is_entity: break
        
        if is_entity:
            continue
            
        # Check Pronouns
        token_text = tokens_list[idx].replace('Ġ', '').lower()
        if token_text in PRONOUNS:
            type_labels[idx] = TYPE_ENT
            # Do not add to token_to_entity (no unification for pronouns)
            continue
            
        # Check Relation Overlap
        for (r_start, r_end, r_type) in relation_spans:
            if max(start, r_start) < min(end, r_end):
                type_labels[idx] = r_type
                break
                
    # 5. Generate Unification Matrix
    unification_matrix = torch.eye(seq_len, dtype=torch.float)
    
    # Group tokens by entity
    entity_to_tokens = {}
    for idx, name in token_to_entity.items():
        if name not in entity_to_tokens:
            entity_to_tokens[name] = []
        entity_to_tokens[name].append(idx)
        
    # Fill matrix
    for name, tokens in entity_to_tokens.items():
        # Create meshgrid of indices
        for i in tokens:
            for j in tokens:
                unification_matrix[i, j] = 1.0
                
    return {
        'input_ids': encoding['input_ids'], # [1, Seq]
        'attention_mask': encoding['attention_mask'], # [1, Seq]
        'type_labels': type_labels, # [Seq]
        'unification_matrix': unification_matrix, # [Seq, Seq]
        'entity_to_tokens': entity_to_tokens # Dict {name: [token_indices]}
    }
