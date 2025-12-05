"""
DATA SPLIT VERIFICATION SCRIPT
==============================
Goal: Mathematically prove that no Subject ID leaks between
Train, Validation, and Test sets.
"""

import random

# 1. The Function we are testing (Copy-pasted from your training scripts)
def get_consistent_split(all_ids, val_ratio=0.15, test_ratio=0.15):
    rng = random.Random(42)  # The Seed of Truth
    ids_copy = all_ids.copy()
    rng.shuffle(ids_copy)
    
    n_total = len(ids_copy)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    
    test_ids = ids_copy[:n_test]
    val_ids = ids_copy[n_test : n_test+n_val]
    train_ids = ids_copy[n_test+n_val:]
    
    return train_ids, val_ids, test_ids

# 2. The Verification Logic
if __name__ == "__main__":
    ALL_SUBJECTS = list(range(1, 110))
    train, val, test = get_consistent_split(ALL_SUBJECTS)

    print(f"Total Subjects: {len(ALL_SUBJECTS)}")
    print(f"Train Size: {len(train)}")
    print(f"Val Size:   {len(val)}")
    print(f"Test Size:  {len(test)}")
    print("-" * 30)

    # CHECK 1: Overlaps
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)

    train_test_overlap = train_set.intersection(test_set)
    train_val_overlap = train_set.intersection(val_set)
    val_test_overlap = val_set.intersection(test_set)

    if len(train_test_overlap) == 0:
        print("✅ PASS: No leakage between Train and Test.")
    else:
        print(f"❌ FAIL: Leakage found! {train_test_overlap}")

    if len(train_val_overlap) == 0:
        print("✅ PASS: No leakage between Train and Val.")
    else:
        print(f"❌ FAIL: Leakage found! {train_val_overlap}")

    if len(val_test_overlap) == 0:
        print("✅ PASS: No leakage between Val and Test.")
    else:
        print(f"❌ FAIL: Leakage found! {val_test_overlap}")

    # CHECK 2: Completeness
    combined_len = len(train) + len(val) + len(test)
    if combined_len == len(ALL_SUBJECTS):
        print(f"✅ PASS: All {combined_len} subjects are accounted for.")
    else:
        print(f"❌ FAIL: Missing subjects! Got {combined_len}, expected {len(ALL_SUBJECTS)}")
        
    print("-" * 30)
    print("Test Subjects (Save this list for your thesis):")
    print(sorted(test))