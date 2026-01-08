
import sys
import os
import re
import csv
from typing import List
from transformers import RobertaTokenizerFast
from tqdm import tqdm
from collections import Counter

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

try:
    from baseline_roberta_analysis import CLUTRRDataset
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from baseline_roberta_analysis import CLUTRRDataset

from clutrr.entity_alignment import align_entity_spans_to_tokens

def get_unique_names_from_brackets(text: str) -> List[str]:
    """Extracts unique names enclosed in square brackets from text."""
    return list(set(re.findall(r"\[(.*?)\]", text)))

def main():
    print("Initializing components...")
    
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    root_dir = "data/clutrr"
    dataset_name = "data_089907f8"
    
    print(f"Loading dataset from {root_dir}/{dataset_name}...")
    dataset = CLUTRRDataset(root=root_dir, dataset=dataset_name, split="train", data_percentage=100)
    
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    limit = 100 
    total_entities = 0
    aligned_entities = 0
    
    span_lengths = []
    
    decoded_checks_passed = 0
    decoded_checks_total = 0
    max_decode_checks = 20
    
    print(f"Verifying alignment on first {limit} samples...")
    print(f"{'SampleID':<10} | {'Entity':<15} | {'Span':<10} | {'Decoded Text':<30} | {'Match?'}")
    print("-" * 85)
    
    for i in tqdm(range(min(limit, len(dataset)))):
        row = dataset.data[i]
        story_text = row[2] 
        
        target_entities = get_unique_names_from_brackets(story_text)
        
        if not target_entities:
            continue

        alignment_result = align_entity_spans_to_tokens(
            text=story_text,
            entities=target_entities,
            tokenizer=tokenizer
        )
        
        input_ids = tokenizer(story_text, return_tensors='pt', add_special_tokens=True)['input_ids'][0]
        
        for entity_name in target_entities:
            total_entities += 1
            result = alignment_result.get(entity_name)
            
            if result and result.get('token_span') is not None:
                start, end = result['token_span']
                
                if start < end: # valid
                    aligned_entities += 1
                    length = end - start
                    span_lengths.append(length)
                    
                    if decoded_checks_total < max_decode_checks:
                        token_ids = input_ids[start:end]
                        decoded_str = tokenizer.decode(token_ids)
                        clean_decoded = decoded_str.strip().replace("Ġ", "")
                        clean_name = entity_name.strip()
                        match = clean_name.lower() in clean_decoded.lower() or clean_decoded.lower() in clean_name.lower()
                        status = "PASS" if match else "FAIL"
                        if match: 
                            decoded_checks_passed += 1
                        print(f"{i:<10} | {entity_name:<15} | {str((start, end)):<10} | {repr(decoded_str):<30} | {status}")
                        decoded_checks_total += 1

    print("\n" + "="*30)
    print("FINAL ALIGNMENT STATISTICS")
    print("="*30)
    print(f"1. Total Entities (N): {total_entities}")
    print(f"2. Successful Hits (N_hit): {aligned_entities}")
    if total_entities > 0:
        print(f"   Hit Rate: {aligned_entities/total_entities*100:.2f}%")
    
    len_counts = Counter(span_lengths)
    sorted_lens = sorted(len_counts.items())
    print("\n3. Span Length Distribution:")
    for length, count in sorted_lens:
        print(f"   Length {length}: {count} occurrences ({count/len(span_lengths)*100:.1f}%)")
        
    print(f"\n4. Decoding Check (Sample {max_decode_checks}):")
    print(f"   Passed: {decoded_checks_passed}/{decoded_checks_total}")
    if decoded_checks_total > 0:
        print(f"   Pass Rate: {decoded_checks_passed/decoded_checks_total*100:.2f}%")
        
    if total_entities > 0 and aligned_entities/total_entities > 0.95 and decoded_checks_passed == decoded_checks_total:
         print("\n✅ Verification PASSED")
    else:
         print("\n❌ Verification FAILED or WARNING")

if __name__ == "__main__":
    main()
