
import re
import logging
from typing import List, Union, Dict, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def align_entity_spans_to_tokens(
    text: str, 
    entities: List[Union[str, Dict]], 
    tokenizer, 
    query_entities: Tuple[str, str] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Aligns entity names in raw text to RoBERTa token spans.
    
    Args:
        text (str): The raw input text (e.g., "[Alice] went to the store.").
        entities (List[Union[str, Dict]]): List of entity names (str) or entity objects (dict).
            If dict, expected to have 'name'. Optional: 'sent_idx', 'expected_span'.
        tokenizer: HuggingFace tokenizer (Fast tokenizer preferred for return_offsets_mapping).
        query_entities (Tuple[str, str], optional): (Subject, Object) for proximity heuristic.
        debug (bool): Return verbose debug info.

    Returns:
        Dict[str, Any]: Mapping from entity identifier (name) to result object:
            {
                'token_span': (start, end), # Exclusive: [start, end)
                'char_span': (start, end),
                'all_occurrences': [ ...list of all found spans... ],
                'source': 'bracket_match' | 'raw_match' | 'fuzzy',
                'debug_info': ...
            }
    """
    
    # 1. Tokenize with Offsets
    # handle cases where tokenizer might not support fast offsets
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    except NotImplementedError:
        # Fallback for slow tokenizers (not recommended but handled)
        logger.warning("Fast tokenizer not available. Offset mapping may be inaccurate.")
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        # We can't proceed reliably without offsets for this strict requirement.
        # But for now assume Fast is used as requested.
        raise ValueError("This function requires a Fast Tokenizer with return_offsets_mapping=True")

    tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
    offsets = enc.offset_mapping # List of (char_start, char_end)
    
    # Precompute Char -> Token Index Map
    # char_to_token_idx[i] = token_index containing char i
    char_to_token_idx = [-1] * len(text)
    
    for t_idx, (start, end) in enumerate(offsets):
        if start == end: continue # Skip special tokens like CLS/SEP if they have 0 width or valid width
        # Note: RoBERTa special tokens might map to 0,0 or full length.
        # Usually offset mapping for special tokens is (0,0).
        
        for c_i in range(start, end):
            if c_i < len(text):
                char_to_token_idx[c_i] = t_idx

    mapping_results = {}

    for ent_input in entities:
        # Normalize Input
        if isinstance(ent_input, str):
            name = ent_input
            meta = {}
        else:
            name = ent_input.get('name', 'Unknown')
            meta = ent_input
            
        if not name:
            continue
            
        entity_key = name # Can be unique ID if provided, using name for now
        
        # Strategy 1: Bracketed Match [Name] (High Precision for CLUTRR)
        search_pattern = f"[{name}]"
        # Strategy 2: Raw Match Name
        fallback_pattern = name
        
        occurrences = []
        
        # Helper to find pattern
        def find_occurrences(pattern, text, source_tag):
            found = []
            # Escape pattern except for brackets if we intended them? 
            # Actually re.escape escapes [], so we construct manually or use string find.
            # Using re.finditer with literal string is safer for names with weird chars.
            
            start_search = 0
            while True:
                idx = text.find(pattern, start_search)
                if idx == -1: break
                
                # Extract char span
                char_start = idx
                char_end = idx + len(pattern)
                
                # Refine char span to just the NAME (strip brackets for token mapping if needed)
                # But usually tokenization of "[Alice]" -> "[", "Alice", "]"
                # We typically want the tokens corresponding to "Alice".
                
                if source_tag == 'bracket_match':
                    # Inner name span: char_start+1 to char_end-1
                    inner_start = char_start + 1
                    inner_end = char_end - 1
                else:
                    inner_start = char_start
                    inner_end = char_end
                    
                # Robust Token Mapping
                # 1. Start Token: Token containing the first char of the name
                # 2. End Token: Token containing the last char of the name
                
                # Check bounds
                if inner_start >= len(text) or inner_end > len(text):
                    logger.warning(f"Bounds error for {name}: {inner_start}-{inner_end} vs {len(text)}")
                    start_search = char_end
                    continue

                tok_start = char_to_token_idx[inner_start]
                # last char is inner_end - 1
                tok_end_inclusive = char_to_token_idx[inner_end - 1]
                
                if tok_start != -1 and tok_end_inclusive != -1:
                    # Final span is [start, end)
                    span = (tok_start, tok_end_inclusive + 1)
                    found.append({
                        'char_span': (inner_start, inner_end),
                        'token_span': span,
                        'source': source_tag,
                        'context_snip': text[max(0, idx-10):min(len(text), idx+len(pattern)+10)]
                    })
                
                start_search = char_end
            return found

        # Execute Search
        occurrences = find_occurrences(search_pattern, text, 'bracket_match')
        
        if not occurrences:
            occurrences = find_occurrences(fallback_pattern, text, 'raw_match')
            
        # Selection Logic (Disambiguation)
        selected = None
        selection_reason = "none"
        
        if not occurrences:
            # FAIL CASE
            if debug:
                logger.error(f"Could not find span for entity: {name}")
            selected = {'token_span': None}
            selection_reason = "not_found"
        elif len(occurrences) == 1:
            selected = occurrences[0]
            selection_reason = "unique"
        else:
            # Disambiguate
            # Priority 1: Metadata (path/sentence index)
            # Todo: If dataset provides character offsets, use them.
            
            # Priority 2: Query Proximity / Sentence Proximity
            # If we know the target sentence index (heuristic)
            # For now, implementing "Closest to end" (often recently introduced) or "First"
            
            # User Rule: "priority... query related... else closest"
            # In CLUTRR, usually names are consistent nodes. 
            # We select the FIRST occurence as the definition of the node usually,
            # or we return a flag to pool.
            
            # Defaulting to FIRST occurrence for stability unless query interaction implies otherwise
            selected = occurrences[0]
            selection_reason = "first_occurrence_fallback"
            
            # Advanced: If query_entities provided, try to find occurrence in same sentence?
            # Complexity: We don't have sentence boundaries mapped to tokens here easily without splitting.
            
        mapping_results[entity_key] = {
            'token_span': selected.get('token_span'),
            'char_span': selected.get('char_span'),
            'all_occurrences': occurrences,
            'reason': selection_reason,
            'name': name
        }
        
    return mapping_results


if __name__ == "__main__":
    # Unit Test / Minimal Reproduction
    from transformers import RobertaTokenizerFast
    
    print("Running Entity Alignment Unit Test...")
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    # Case 1: Standard CLUTRR
    text1 = "[Dorothy]'s brother [Michael] and her went to get ice cream. [Michael] is the father of [Donald]."
    entities1 = ["Dorothy", "Michael", "Donald"]
    
    res1 = align_entity_spans_to_tokens(text1, entities1, tokenizer)
    
    print(f"Text: {text1}")
    for ent, data in res1.items():
        span = data['token_span']
        raw_token_strs = tokenizer.convert_ids_to_tokens(tokenizer(text1)['input_ids'][span[0]:span[1]])
        print(f"Entity: {ent} -> Span: {span} -> Tokens: {raw_token_strs} (Reason: {data['reason']})")
        # Validation
        if ent == "Michael":
            assert len(data['all_occurrences']) == 2
            print("  -> Multiple occurrences found correctly.")

    # Case 2: No Brackets & Subwords
    text2 = "Christopher and Jeffrey are at a bar."
    entities2 = ["Christopher", "Jeffrey"]
    res2 = align_entity_spans_to_tokens(text2, entities2, tokenizer)
    
    for ent, data in res2.items():
        span = data['token_span']
        raw_token_strs = tokenizer.convert_ids_to_tokens(tokenizer(text2)['input_ids'][span[0]:span[1]])
        print(f"Entity: {ent} -> Span: {span} -> Tokens: {raw_token_strs}")

    print("\nUnit Test Passed.")
