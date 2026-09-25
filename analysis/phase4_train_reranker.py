#!/usr/bin/env python3
"""
Phase 4 Step 2: Training Data Extraction & ML Reranker Training
Amazon ML Challenge 2026 — Business Entity Resolution

This module implements the end-to-end Phase 4 Stage 2 ML pipeline:
1. Deterministically selects 30,000 Source 1 training entities (seed 43, disjoint from val).
2. Loads the frozen 10,000 validation IDs from analysis/phase2_validation_ids.txt.
3. Simultaneously extracts candidate pairs and profiles for training and validation
   in a single streaming pass over Source 2 and Source 3 (< 15 min, < 350 MB RAM).
4. Constructs training labels (y in {0, 1}) from train_ground_truth.tsv for train IDs only.
5. Builds the 18-dimensional feature matrix for train and validation using phase4_features.
6. Trains HistGradientBoostingClassifier (and LogisticRegression ablation baseline).
7. Evaluates on the frozen 10,000 validation entities with threshold sweep (tau, tau_singleton).
8. Saves the best model artifact to analysis/phase4_model.joblib.
9. Writes the comprehensive performance report to analysis/phase4_validation_report.txt.
"""

import os
import sys
import time
import math
import random
import collections
import tracemalloc
from typing import Dict, Set, List, Tuple, Any, Optional

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ensure analysis directory is on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from phase4_features import (
    normalize_business_name,
    tokenize,
    char_ngrams,
    token_jaccard,
    token_overlap,
    extract_digit_blocks,
    extract_domain_stem,
    build_record_profile,
    extract_feature_vector,
    first_token,
    FEATURE_NAMES,
)
from phase4_candidate_generation import (
    select_training_s1_ids,
    load_s1_records_by_ids,
    first_2_tokens,
    DEFAULT_TRAIN_SAMPLE_SIZE,
    DEFAULT_TRAIN_SEED,
)

# File paths
DATASET_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
TRAIN_S1_PATH = os.path.join(DATASET_TRAIN_DIR, "train_source1.tsv")
TRAIN_S2_PATH = os.path.join(DATASET_TRAIN_DIR, "train_source2.tsv")
TRAIN_S3_PATH = os.path.join(DATASET_TRAIN_DIR, "train_source3.tsv")
TRAIN_GT_PATH = os.path.join(DATASET_TRAIN_DIR, "train_ground_truth.tsv")

VAL_IDS_PATH = os.path.join(CURRENT_DIR, "phase2_validation_ids.txt")
MODEL_OUTPUT_PATH = os.path.join(CURRENT_DIR, "phase4_model.joblib")
REPORT_OUTPUT_PATH = os.path.join(CURRENT_DIR, "phase4_validation_report.txt")

# Phase 3B comparison benchmark
PHASE3B_MACRO_F05 = 0.6008
PHASE3B_MACRO_PREC = 0.6451
PHASE3B_MACRO_REC = 0.5907
PHASE3B_MICRO_PREC = 0.3526
PHASE3B_MICRO_REC = 0.6040
PHASE3B_TP = 20798
PHASE3B_FP = 38182
PHASE3B_FN = 13638
PHASE3B_SINGLETON_ACC = 39.39

# Candidate safety cap per entity per source (e.g. 30 from S2, 30 from S3 = up to 60 total)
MAX_CANDIDATES_PER_SOURCE = 30


def compute_entity_f05(
    pred_set: Set[str],
    true_set: Set[str]
) -> Tuple[float, float, float, int, int, int]:
    """Compute per-entity Precision, Recall, and F0.5 matching official criteria."""
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0, 0, 0, 0
        else:
            return 0.0, 0.0, 0.0, 0, len(pred_set), 0
    else:
        if len(pred_set) == 0:
            return 0.0, 0.0, 0.0, 0, 0, len(true_set)
        else:
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            prec = tp / len(pred_set)
            rec = tp / len(true_set)
            if tp > 0:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            else:
                f05 = 0.0
            return prec, rec, f05, tp, fp, fn


def extract_unified_candidates(
    train_s1_records: Dict[str, Dict[str, str]],
    val_s1_records: Dict[str, Dict[str, str]],
    s2_path: str = TRAIN_S2_PATH,
    s3_path: str = TRAIN_S3_PATH,
    max_s2_rows: Optional[int] = None,
    max_s3_rows: Optional[int] = None
) -> Tuple[
    Dict[str, Dict[str, Any]],               # all_s1_profiles
    Dict[str, Dict[str, Any]],               # cand_profiles
    Dict[str, List[str]],                    # train_candidates: s1_id -> [cand_ids]
    Dict[str, List[str]],                    # val_candidates: s1_id -> [cand_ids]
]:
    """
    Simultaneously extracts candidate pools and precomputed profiles for both
    train and validation entities in a single streaming pass over S2 and S3.
    """
    print("\nBuilding multi-pass blocking indices for Train (30k) and Val (10k)...")
    t0 = time.perf_counter()

    all_s1_profiles: Dict[str, Dict[str, Any]] = {}
    s1_split: Dict[str, str] = {}

    block_exact: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    block_f2: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    block_dom: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
    block_dig: Dict[Tuple[str, str, str], List[str]] = collections.defaultdict(list)

    # Index train records
    for eid, rec in train_s1_records.items():
        p = build_record_profile(rec["name"], rec["addr"], rec["country"])
        all_s1_profiles[eid] = p
        s1_split[eid] = "train"
        n, country = p["norm_name"], p["country"]
        if n:
            block_exact[(n, country)].append(eid)
        f2 = first_2_tokens(n)
        if len(f2) >= 4:
            block_f2[(f2, country)].append(eid)
        stem = p["stripped_name"]
        if len(stem) >= 4:
            block_dom[(stem, country)].append(eid)
        f1 = p["first_tok"]
        if len(f1) >= 4 and p["digits"]:
            for d in p["digits"]:
                block_dig[(d, f1, country)].append(eid)

    # Index val records
    for eid, rec in val_s1_records.items():
        p = build_record_profile(rec["name"], rec["addr"], rec["country"])
        all_s1_profiles[eid] = p
        s1_split[eid] = "val"
        n, country = p["norm_name"], p["country"]
        if n:
            block_exact[(n, country)].append(eid)
        f2 = first_2_tokens(n)
        if len(f2) >= 4:
            block_f2[(f2, country)].append(eid)
        stem = p["stripped_name"]
        if len(stem) >= 4:
            block_dom[(stem, country)].append(eid)
        f1 = p["first_tok"]
        if len(f1) >= 4 and p["digits"]:
            for d in p["digits"]:
                block_dig[(d, f1, country)].append(eid)

    print(f"Indexing completed in {time.perf_counter() - t0:.2f}s across {len(all_s1_profiles):,} S1 entities.")
    print(f"  - Exact name blocks     : {len(block_exact):,}")
    print(f"  - First-2-tokens blocks  : {len(block_f2):,}")
    print(f"  - Domain stem blocks    : {len(block_dom):,}")
    print(f"  - Digit+token blocks    : {len(block_dig):,}")

    # Streaming candidate collection
    train_cands_raw: Dict[str, Dict[str, Set[str]]] = {eid: {"s2": set(), "s3": set()} for eid in train_s1_records}
    val_cands_raw: Dict[str, Dict[str, Set[str]]] = {eid: {"s2": set(), "s3": set()} for eid in val_s1_records}
    cand_profiles: Dict[str, Dict[str, Any]] = {}

    sources = [("S2", s2_path, max_s2_rows), ("S3", s3_path, max_s3_rows)]

    for src_label, src_path, max_rows in sources:
        t_src = time.perf_counter()
        scanned = 0
        hits = 0
        src_key = src_label.lower()
        print(f"\nStreaming {src_label} from {os.path.relpath(src_path)}...")

        with open(src_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
            f.readline()  # Skip header
            for line in f:
                if max_rows and scanned >= max_rows:
                    break
                scanned += 1
                if scanned % 1000000 == 0:
                    elapsed = time.perf_counter() - t_src
                    print(f"  [{src_label}] Processed {scanned:,} rows ({scanned/elapsed:,.0f} rows/s, {hits:,} candidates captured)...")

                parts = line.rstrip('\r\n').split('\t')
                if len(parts) >= 4:
                    eid = parts[0].strip()
                    bname = parts[1]
                    baddr = parts[2]
                    country = parts[3].strip()

                    n = normalize_business_name(bname)

                    k_ex = (n, country) if n else None
                    f2 = first_2_tokens(n)
                    k_f2 = (f2, country) if len(f2) >= 4 else None
                    dom_s = extract_domain_stem(bname)
                    k_dom = (dom_s, country) if (dom_s and len(dom_s) >= 4) else None

                    # Matching S1 entities for this candidate
                    matched_s1_for_cand: Set[str] = set()

                    # Pass 1: Exact Name
                    if k_ex and k_ex in block_exact:
                        for s1_id in block_exact[k_ex]:
                            c_dict = train_cands_raw if s1_split[s1_id] == "train" else val_cands_raw
                            if len(c_dict[s1_id][src_key]) < MAX_CANDIDATES_PER_SOURCE:
                                c_dict[s1_id][src_key].add(eid)
                                matched_s1_for_cand.add(s1_id)

                    # Pass 2: Domain Stem
                    if k_dom and k_dom in block_dom:
                        for s1_id in block_dom[k_dom]:
                            c_dict = train_cands_raw if s1_split[s1_id] == "train" else val_cands_raw
                            if len(c_dict[s1_id][src_key]) < MAX_CANDIDATES_PER_SOURCE:
                                c_dict[s1_id][src_key].add(eid)
                                matched_s1_for_cand.add(s1_id)

                    # Pass 3: First-2-Tokens Prefix Match (Fuzzy Name)
                    if k_f2 and k_f2 in block_f2:
                        t_toks = None
                        t_ngs = None
                        for s1_id in block_f2[k_f2]:
                            c_dict = train_cands_raw if s1_split[s1_id] == "train" else val_cands_raw
                            if eid not in c_dict[s1_id][src_key] and len(c_dict[s1_id][src_key]) < MAX_CANDIDATES_PER_SOURCE:
                                if t_toks is None:
                                    t_toks = tokenize(bname)
                                    t_ngs = char_ngrams(n, 3)
                                s1 = all_s1_profiles[s1_id]
                                tj = token_jaccard(s1["name_tokens"], t_toks)
                                gj = token_jaccard(s1["name_ngrams"], t_ngs)
                                if tj >= 0.40 or gj >= 0.45:
                                    c_dict[s1_id][src_key].add(eid)
                                    matched_s1_for_cand.add(s1_id)

                    # Pass 4: Digit Block + First Token Match
                    f1 = first_token(n)
                    if len(f1) >= 4:
                        t_digs = extract_digit_blocks(baddr)
                        if t_digs:
                            t_toks_d = None
                            for d in t_digs:
                                k_dig = (d, f1, country)
                                if k_dig in block_dig:
                                    for s1_id in block_dig[k_dig]:
                                        c_dict = train_cands_raw if s1_split[s1_id] == "train" else val_cands_raw
                                        if eid not in c_dict[s1_id][src_key] and len(c_dict[s1_id][src_key]) < MAX_CANDIDATES_PER_SOURCE:
                                            if t_toks_d is None:
                                                t_toks_d = tokenize(bname)
                                            s1 = all_s1_profiles[s1_id]
                                            if token_overlap(s1["name_tokens"], t_toks_d) >= 1:
                                                c_dict[s1_id][src_key].add(eid)
                                                matched_s1_for_cand.add(s1_id)

                    if matched_s1_for_cand:
                        hits += len(matched_s1_for_cand)
                        if eid not in cand_profiles:
                            cand_profiles[eid] = build_record_profile(bname, baddr, country)

        print(f"Finished {src_label} in {time.perf_counter() - t_src:.2f}s ({scanned:,} rows scanned, {hits:,} candidate links).")

    # Consolidate candidate lists
    train_candidates: Dict[str, List[str]] = {}
    for s1_id, pool in train_cands_raw.items():
        train_candidates[s1_id] = sorted(pool["s2"]) + sorted(pool["s3"])

    val_candidates: Dict[str, List[str]] = {}
    for s1_id, pool in val_cands_raw.items():
        val_candidates[s1_id] = sorted(pool["s2"]) + sorted(pool["s3"])

    return all_s1_profiles, cand_profiles, train_candidates, val_candidates


def build_feature_matrix_and_labels(
    s1_ids: List[str],
    s1_profiles: Dict[str, Dict[str, Any]],
    cand_profiles: Dict[str, Dict[str, Any]],
    candidates: Dict[str, List[str]],
    ground_truth: Optional[Dict[str, Set[str]]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[str, str]]]:
    """
    Builds the 18-dimensional feature matrix X and label vector y for candidate pairs.
    """
    X_rows = []
    y_vals = [] if ground_truth is not None else None
    pair_meta = []

    for s1_id in s1_ids:
        c_list = candidates.get(s1_id, [])
        cand_pool_size = len(c_list)
        s1_prof = s1_profiles[s1_id]
        true_matches = ground_truth.get(s1_id, set()) if ground_truth is not None else None

        for cid in c_list:
            cand_prof = cand_profiles[cid]
            is_s3 = cid.startswith("S3-")

            feat_vec = extract_feature_vector(
                rec1=s1_prof,
                rec2=cand_prof,
                is_source3=is_s3,
                cand_pool_size=cand_pool_size
            )
            X_rows.append(feat_vec)
            pair_meta.append((s1_id, cid))

            if y_vals is not None:
                label = 1 if cid in true_matches else 0
                y_vals.append(label)

    X = np.array(X_rows, dtype=np.float32) if X_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.array(y_vals, dtype=np.int32) if y_vals is not None else None
    return X, y, pair_meta


def evaluate_threshold_sweep(
    val_ids: List[str],
    pair_meta: List[Tuple[str, str]],
    y_probs: np.ndarray,
    val_gt: Dict[str, Set[str]],
    tau_range: np.ndarray,
    tau_singleton_range: np.ndarray
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Sweeps tau and tau_singleton combinations to maximize Macro F0.5 on validation.
    """
    # Group probabilities by s1_id
    preds_by_s1: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for (s1_id, cid), prob in zip(pair_meta, y_probs):
        preds_by_s1[s1_id].append((cid, float(prob)))

    best_macro_f05 = -1.0
    best_tau = 0.50
    best_tau_singleton = 0.50
    best_metrics: Dict[str, Any] = {}

    total_entities = len(val_ids)

    for tau_sing in tau_singleton_range:
        for tau in tau_range:
            tot_tp = 0
            tot_fp = 0
            tot_fn = 0
            f05_list = []
            prec_list = []
            rec_list = []
            singletons_correct = 0
            singletons_true = 0

            for s1_id in val_ids:
                true_set = val_gt.get(s1_id, set())
                cand_pairs = preds_by_s1.get(s1_id, [])

                # Singleton determination
                if len(true_set) == 0:
                    singletons_true += 1

                # Decision policy
                if not cand_pairs:
                    pred_set: Set[str] = set()
                else:
                    max_prob = max(prob for _, prob in cand_pairs)
                    if max_prob < tau_sing:
                        # Singleton Guard triggers: emit empty set
                        pred_set = set()
                    else:
                        pred_set = {cid for cid, prob in cand_pairs if prob >= tau}

                if len(true_set) == 0 and len(pred_set) == 0:
                    singletons_correct += 1

                p, r, f, tp, fp, fn = compute_entity_f05(pred_set, true_set)
                prec_list.append(p)
                rec_list.append(r)
                f05_list.append(f)
                tot_tp += tp
                tot_fp += fp
                tot_fn += fn

            macro_f05 = sum(f05_list) / total_entities
            macro_prec = sum(prec_list) / total_entities
            macro_rec = sum(rec_list) / total_entities

            micro_prec = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
            micro_rec = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
            singleton_acc = (singletons_correct / singletons_true * 100.0) if singletons_true > 0 else 0.0

            if macro_f05 > best_macro_f05:
                best_macro_f05 = macro_f05
                best_tau = float(tau)
                best_tau_singleton = float(tau_sing)
                best_metrics = {
                    "macro_f05": macro_f05,
                    "macro_precision": macro_prec,
                    "macro_recall": macro_rec,
                    "micro_precision": micro_prec,
                    "micro_recall": micro_rec,
                    "tp": tot_tp,
                    "fp": tot_fp,
                    "fn": tot_fn,
                    "singletons_correct": singletons_correct,
                    "singletons_true": singletons_true,
                    "singleton_accuracy": singleton_acc,
                }

    return best_tau, best_tau_singleton, best_macro_f05, best_metrics


def main():
    print("=" * 80)
    print("Starting Amazon ML Challenge 2026 — Phase 4 Step 2: ML Reranker Pipeline")
    print("=" * 80)

    tracemalloc.start()
    pipeline_start = time.perf_counter()

    # Step 1: Select deterministic training S1 IDs (seed 43, 30,000 entities)
    print("\n--- STEP 2A: Training Sample Selection ---")
    train_ids = select_training_s1_ids(
        s1_path=TRAIN_S1_PATH,
        val_ids_path=VAL_IDS_PATH,
        sample_size=DEFAULT_TRAIN_SAMPLE_SIZE,
        seed=DEFAULT_TRAIN_SEED
    )

    # Step 2: Load frozen validation S1 IDs
    with open(VAL_IDS_PATH, 'r', encoding='utf-8') as f:
        val_ids = [line.strip() for line in f if line.strip()]

    # Verify disjointness
    overlap = set(train_ids) & set(val_ids)
    assert len(overlap) == 0, f"FATAL: Train and Val overlap detected: {len(overlap)}"
    print(f"Train IDs: {len(train_ids):,} | Val IDs: {len(val_ids):,} | Overlap: {len(overlap)}")

    # Step 3: Load S1 text records
    print("\nLoading Source 1 records for Train and Val...")
    train_s1_records = load_s1_records_by_ids(TRAIN_S1_PATH, set(train_ids))
    val_s1_records = load_s1_records_by_ids(TRAIN_S1_PATH, set(val_ids))
    print(f"Loaded {len(train_s1_records):,} train S1 records and {len(val_s1_records):,} val S1 records.")

    # Step 4: Unified Streaming Candidate Generation
    print("\n--- STEP 2B: Unified Candidate Extraction (Train + Val) ---")
    t_cand_start = time.perf_counter()
    all_s1_profiles, cand_profiles, train_cands, val_cands = extract_unified_candidates(
        train_s1_records=train_s1_records,
        val_s1_records=val_s1_records,
        s2_path=TRAIN_S2_PATH,
        s3_path=TRAIN_S3_PATH
    )
    cand_gen_time = time.perf_counter() - t_cand_start
    print(f"\nCandidate extraction complete in {cand_gen_time:.2f}s.")
    print(f"Unique candidate records cached: {len(cand_profiles):,}")

    train_cand_count = sum(len(c) for c in train_cands.values())
    val_cand_count = sum(len(c) for c in val_cands.values())
    print(f"Train candidate pairs: {train_cand_count:,} ({train_cand_count/len(train_ids):.1f} avg/entity)")
    print(f"Val candidate pairs  : {val_cand_count:,} ({val_cand_count/len(val_ids):.1f} avg/entity)")

    # Step 5: Load Ground Truth for Train entities ONLY
    print("\n--- STEP 2C: Ground Truth Loading for Training ---")
    train_ids_set = set(train_ids)
    train_gt: Dict[str, Set[str]] = {eid: set() for eid in train_ids}
    with open(TRAIN_GT_PATH, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            eid = parts[0]
            if eid in train_ids_set:
                matched_str = parts[1].strip() if len(parts) > 1 else ""
                if matched_str:
                    for mid in matched_str.split(','):
                        m = mid.strip()
                        if m:
                            train_gt[eid].add(m)

    total_train_gt_matches = sum(len(m) for m in train_gt.values())
    train_singletons = sum(1 for m in train_gt.values() if len(m) == 0)
    print(f"Train GT total matches: {total_train_gt_matches:,}")
    print(f"Train true singletons : {train_singletons:,} ({train_singletons/len(train_ids)*100:.2f}%)")

    # Step 6: Feature Matrix & Label Construction
    print("\n--- STEP 2D: Feature Extraction (18 dimensions) ---")
    t_feat = time.perf_counter()
    X_train, y_train, train_meta = build_feature_matrix_and_labels(
        s1_ids=train_ids,
        s1_profiles=all_s1_profiles,
        cand_profiles=cand_profiles,
        candidates=train_cands,
        ground_truth=train_gt
    )
    X_val, _, val_meta = build_feature_matrix_and_labels(
        s1_ids=val_ids,
        s1_profiles=all_s1_profiles,
        cand_profiles=cand_profiles,
        candidates=val_cands,
        ground_truth=None  # Zero ground truth during validation feature extraction!
    )
    feat_time = time.perf_counter() - t_feat
    print(f"Feature extraction complete in {feat_time:.2f}s.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape  : {X_val.shape}")

    # Feature verification
    assert X_train.shape[1] == 18, f"Expected 18 features, got {X_train.shape[1]}"
    assert X_val.shape[1] == 18, f"Expected 18 features, got {X_val.shape[1]}"
    assert not np.isnan(X_train).any(), "NaN detected in X_train"
    assert not np.isinf(X_train).any(), "Inf detected in X_train"

    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    pos_neg_ratio = (pos_count / neg_count) if neg_count > 0 else 0.0
    cand_recall = (pos_count / total_train_gt_matches) if total_train_gt_matches > 0 else 0.0

    print(f"Train labels summary:")
    print(f"  * Positive pairs (y=1) : {pos_count:,}")
    print(f"  * Negative pairs (y=0) : {neg_count:,}")
    print(f"  * Positive/Negative ratio: {pos_neg_ratio:.4f}")
    print(f"  * Candidate Recall ceiling: {pos_count:,} / {total_train_gt_matches:,} ({cand_recall*100:.2f}%)")

    # Free candidate profiles memory
    del cand_profiles

    # Step 7: Model Training (HistGradientBoostingClassifier & LogisticRegression Ablation)
    print("\n--- STEP 2E: Model Training ---")
    t_train = time.perf_counter()
    print("Training HistGradientBoostingClassifier (random_state=42)...")
    hgb_model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=150,
        learning_rate=0.08,
        min_samples_leaf=20,
        max_leaf_nodes=31
    )
    hgb_model.fit(X_train, y_train)
    hgb_train_time = time.perf_counter() - t_train
    print(f"HistGradientBoostingClassifier trained in {hgb_train_time:.2f}s.")

    # Train LogisticRegression for ablation
    print("Training LogisticRegression ablation baseline...")
    t_lr = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_train_time = time.perf_counter() - t_lr
    print(f"LogisticRegression trained in {lr_train_time:.2f}s.")

    # Step 8: Load Validation Ground Truth (Post-training scoring ONLY)
    print("\n--- STEP 2F: Loading Validation Ground Truth for Evaluation ---")
    val_ids_set = set(val_ids)
    val_gt: Dict[str, Set[str]] = {eid: set() for eid in val_ids}
    with open(TRAIN_GT_PATH, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            eid = parts[0]
            if eid in val_ids_set:
                matched_str = parts[1].strip() if len(parts) > 1 else ""
                if matched_str:
                    for mid in matched_str.split(','):
                        m = mid.strip()
                        if m:
                            val_gt[eid].add(m)

    total_val_gt_matches = sum(len(m) for m in val_gt.values())
    val_singletons = sum(1 for m in val_gt.values() if len(m) == 0)
    print(f"Validation GT matches: {total_val_gt_matches:,} across {len(val_ids):,} entities.")
    print(f"Validation singletons: {val_singletons:,} ({val_singletons/len(val_ids)*100:.2f}%)")

    # Step 9: Validation Inference and Threshold Sweep
    print("\n--- STEP 2G: Validation Inference & Threshold Sweep ---")
    t_val_inf = time.perf_counter()
    hgb_val_probs = hgb_model.predict_proba(X_val)[:, 1]
    X_val_scaled = scaler.transform(X_val)
    lr_val_probs = lr_model.predict_proba(X_val_scaled)[:, 1]

    tau_range = np.arange(0.20, 0.81, 0.02)
    tau_singleton_range = np.arange(0.30, 0.71, 0.05)

    print(f"Sweeping {len(tau_range)} tau values x {len(tau_singleton_range)} tau_singleton values (279 combinations)...")
    best_tau, best_tau_sing, best_f05, hgb_metrics = evaluate_threshold_sweep(
        val_ids=val_ids,
        pair_meta=val_meta,
        y_probs=hgb_val_probs,
        val_gt=val_gt,
        tau_range=tau_range,
        tau_singleton_range=tau_singleton_range
    )
    val_eval_time = time.perf_counter() - t_val_inf
    print(f"Validation sweep completed in {val_eval_time:.2f}s.")
    print(f"Best HistGradientBoosting Macro F0.5: {best_f05:.4f} (tau={best_tau:.2f}, tau_singleton={best_tau_sing:.2f})")

    # Evaluate LogisticRegression ablation
    lr_best_tau, lr_best_tau_sing, lr_best_f05, lr_metrics = evaluate_threshold_sweep(
        val_ids=val_ids,
        pair_meta=val_meta,
        y_probs=lr_val_probs,
        val_gt=val_gt,
        tau_range=tau_range,
        tau_singleton_range=tau_singleton_range
    )
    print(f"Best LogisticRegression Macro F0.5    : {lr_best_f05:.4f} (tau={lr_best_tau:.2f}, tau_singleton={lr_best_tau_sing:.2f})")

    # Step 10: Save Trained Model Artifact
    print(f"\nSaving model artifact to {os.path.relpath(MODEL_OUTPUT_PATH)}...")
    model_artifact = {
        "model": hgb_model,
        "feature_names": FEATURE_NAMES,
        "selected_tau": best_tau,
        "selected_tau_singleton": best_tau_sing,
        "val_macro_f05": best_f05,
        "val_metrics": hgb_metrics,
        "lr_model": lr_model,
        "scaler": scaler,
        "lr_metrics": lr_metrics,
        "train_candidate_count": train_cand_count,
        "train_pos_labels": pos_count,
        "train_neg_labels": neg_count,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }
    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)
    print("Model artifact saved successfully.")

    # Step 11: Write Validation Report
    delta_f05 = best_f05 - PHASE3B_MACRO_F05
    delta_prec = hgb_metrics["macro_precision"] - PHASE3B_MACRO_PREC
    delta_rec = hgb_metrics["macro_recall"] - PHASE3B_MACRO_REC

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = time.perf_counter() - pipeline_start

    hgb_sing_acc_str = f"{hgb_metrics['singleton_accuracy']:.2f}%"
    lr_sing_acc_str = f"{lr_metrics['singleton_accuracy']:.2f}%"
    p3b_sing_acc_str = f"{PHASE3B_SINGLETON_ACC:.2f}%"
    hgb_train_time_str = f"{hgb_train_time:.2f}s"
    lr_train_time_str = f"{lr_train_time:.2f}s"

    report_lines = [
        "=" * 80,
        "AMAZON ML CHALLENGE 2026 — BUSINESS ENTITY RESOLUTION",
        "PHASE 4 STEP 2: ML RERANKER TRAINING & VALIDATION REPORT",
        "=" * 80,
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Training Sample Size   : {len(train_ids):,} Source 1 Entities (Seed=43, Disjoint from Val)",
        f"Validation Sample Size : {len(val_ids):,} Source 1 Entities (Seed=42, Frozen)",
        f"Comparison Baseline    : Phase 3B Heuristic Pipeline (Macro F0.5 = {PHASE3B_MACRO_F05:.4f})",
        "-" * 80,
        "EXECUTIVE PERFORMANCE HEADLINE:",
        f"  * Phase 4 Macro F0.5     : {best_f05:.4f}  (Phase 3B: {PHASE3B_MACRO_F05:.4f} | Delta: {delta_f05:+.4f})",
        f"  * Phase 4 Macro Precision: {hgb_metrics['macro_precision']:.4f}  (Phase 3B: {PHASE3B_MACRO_PREC:.4f} | Delta: {delta_prec:+.4f})",
        f"  * Phase 4 Macro Recall   : {hgb_metrics['macro_recall']:.4f}  (Phase 3B: {PHASE3B_MACRO_REC:.4f} | Delta: {delta_rec:+.4f})",
        f"  * Optimal tau (match)    : {best_tau:.2f}",
        f"  * Optimal tau_singleton  : {best_tau_sing:.2f}",
        f"  * Singleton Accuracy     : {hgb_metrics['singletons_correct']}/{hgb_metrics['singletons_true']} ({hgb_sing_acc_str} vs Phase 3B {p3b_sing_acc_str})",
        "=" * 80,
        "",
        "SECTION 1: HEAD-TO-HEAD COMPARISON AGAINST PHASE 3B",
        "=" * 80,
        f"{'Metric':<34} | {'Phase 3B Heuristic':<20} | {'Phase 4 ML Reranker':<20} | {'Absolute Delta'}",
        "-" * 80,
        f"{'Macro F0.5 (Official Metric)':<34} | {PHASE3B_MACRO_F05:<20.4f} | {best_f05:<20.4f} | {delta_f05:+.4f}",
        f"{'Macro Precision':<34} | {PHASE3B_MACRO_PREC:<20.4f} | {hgb_metrics['macro_precision']:<20.4f} | {delta_prec:+.4f}",
        f"{'Macro Recall':<34} | {PHASE3B_MACRO_REC:<20.4f} | {hgb_metrics['macro_recall']:<20.4f} | {delta_rec:+.4f}",
        f"{'Micro Precision':<34} | {PHASE3B_MICRO_PREC:<20.4f} | {hgb_metrics['micro_precision']:<20.4f} | {hgb_metrics['micro_precision'] - PHASE3B_MICRO_PREC:+.4f}",
        f"{'Micro Recall':<34} | {PHASE3B_MICRO_REC:<20.4f} | {hgb_metrics['micro_recall']:<20.4f} | {hgb_metrics['micro_recall'] - PHASE3B_MICRO_REC:+.4f}",
        f"{'True Positives (TP)':<34} | {PHASE3B_TP:<20,d} | {hgb_metrics['tp']:<20,d} | {hgb_metrics['tp'] - PHASE3B_TP:+,d}",
        f"{'False Positives (FP)':<34} | {PHASE3B_FP:<20,d} | {hgb_metrics['fp']:<20,d} | {hgb_metrics['fp'] - PHASE3B_FP:+,d}",
        f"{'False Negatives (FN)':<34} | {PHASE3B_FN:<20,d} | {hgb_metrics['fn']:<20,d} | {hgb_metrics['fn'] - PHASE3B_FN:+,d}",
        f"{'True Singletons':<34} | {'556':<20} | {hgb_metrics['singletons_true']:<20,d} | {'0'}",
        f"{'Correctly Predicted Singletons':<34} | {'219':<20} | {hgb_metrics['singletons_correct']:<20,d} | {hgb_metrics['singletons_correct'] - 219:+,d}",
        f"{'Singleton Accuracy':<34} | {p3b_sing_acc_str:<20} | {hgb_sing_acc_str:<20} | {hgb_metrics['singleton_accuracy'] - PHASE3B_SINGLETON_ACC:+.2f}%",
        "=" * 80,
        "",
        "SECTION 2: MODEL ABLATION COMPARISON (HistGradientBoosting vs LogisticRegression)",
        "=" * 80,
        f"{'Metric':<34} | {'HistGradientBoosting':<20} | {'LogisticRegression':<20} | {'Difference'}",
        "-" * 80,
        f"{'Macro F0.5':<34} | {best_f05:<20.4f} | {lr_best_f05:<20.4f} | {best_f05 - lr_best_f05:+.4f}",
        f"{'Macro Precision':<34} | {hgb_metrics['macro_precision']:<20.4f} | {lr_metrics['macro_precision']:<20.4f} | {hgb_metrics['macro_precision'] - lr_metrics['macro_precision']:+.4f}",
        f"{'Macro Recall':<34} | {hgb_metrics['macro_recall']:<20.4f} | {lr_metrics['macro_recall']:<20.4f} | {hgb_metrics['macro_recall'] - lr_metrics['macro_recall']:+.4f}",
        f"{'Micro Precision':<34} | {hgb_metrics['micro_precision']:<20.4f} | {lr_metrics['micro_precision']:<20.4f} | {hgb_metrics['micro_precision'] - lr_metrics['micro_precision']:+.4f}",
        f"{'Micro Recall':<34} | {hgb_metrics['micro_recall']:<20.4f} | {lr_metrics['micro_recall']:<20.4f} | {hgb_metrics['micro_recall'] - lr_metrics['micro_recall']:+.4f}",
        f"{'Singleton Accuracy':<34} | {hgb_sing_acc_str:<20} | {lr_sing_acc_str:<20} | {hgb_metrics['singleton_accuracy'] - lr_metrics['singleton_accuracy']:+.2f}%",
        f"{'Selected tau':<34} | {best_tau:<20.2f} | {lr_best_tau:<20.2f} | -",
        f"{'Selected tau_singleton':<34} | {best_tau_sing:<20.2f} | {lr_best_tau_sing:<20.2f} | -",
        f"{'Training Runtime':<34} | {hgb_train_time_str:<20} | {lr_train_time_str:<20} | -",
        "=" * 80,
        "",
        "SECTION 3: TRAINING CANDIDATE & DATASET STATISTICS",
        "=" * 80,
        f"Training S1 Entities Sampled : {len(train_ids):,}",
        f"Validation / Train Overlap   : 0 (Strictly 100% disjoint)",
        f"Total Training Candidate Pairs: {train_cand_count:,}",
        f"  - Positive Labels (y=1)    : {pos_count:,} ({pos_count/train_cand_count*100:.2f}%)",
        f"  - Negative Labels (y=0)    : {neg_count:,} ({neg_count/train_cand_count*100:.2f}%)",
        f"  - Positive/Negative Ratio  : {pos_neg_ratio:.4f} (1 : {1/pos_neg_ratio:.2f})",
        f"  - Candidate Recall Ceiling : {pos_count:,} / {total_train_gt_matches:,} ({cand_recall*100:.2f}%)",
        f"Total Validation Candidates  : {val_cand_count:,}",
        "=" * 80,
        "",
        "SECTION 4: RESOURCE UTILIZATION & RUNTIME AUDIT",
        "=" * 80,
        f"Candidate Generation Runtime : {cand_gen_time:.2f} s",
        f"Feature Extraction Runtime   : {feat_time:.2f} s",
        f"GBDT Model Training Runtime  : {hgb_train_time:.2f} s",
        f"Validation Evaluation Runtime: {val_eval_time:.2f} s",
        f"Total End-to-End Pipeline    : {total_elapsed:.2f} s ({total_elapsed/60:.2f} min)",
        f"Peak Memory Footprint (RAM)  : {peak_mem / (1024*1024):.2f} MB (Budget: 500 MB)",
        "=" * 80,
        "END OF PHASE 4 STEP 2 REPORT"
    ]

    report_text = "\n".join(report_lines)
    with open(REPORT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport written to: {os.path.relpath(REPORT_OUTPUT_PATH)}")
    print(f"Total Pipeline Runtime: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print(f"Peak Memory: {peak_mem / (1024*1024):.2f} MB")
    print("=" * 80)
    print("\n" + report_text)


if __name__ == "__main__":
    main()
