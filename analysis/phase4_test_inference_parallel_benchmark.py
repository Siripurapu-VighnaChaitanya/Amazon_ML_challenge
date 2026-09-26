#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Phase 4 Step 3A: Parallel Indexed Inference Benchmark
=============================================================================
Architectural Experiment:
Investigate whether multiprocessing by country (US, India, France) can reduce
inference time safely and exactly without any correctness loss.

Key Architecture:
- Uses concurrent.futures.ProcessPoolExecutor with 3 workers.
- Each worker processes S1 entities for exactly one country.
- Each worker opens independent read-only SQLite connections to S2 and S3 indexes.
- Exact candidate generation logic, 18 features, model parameters, and
  chronological capping behavior are preserved perfectly.
- Parent process merges the results to preserve the exact S1 test file order.
- Performs byte-for-byte / SHA-256 comparison against the baseline benchmark.
=============================================================================
"""

import collections
import concurrent.futures
import hashlib
import os
import sqlite3
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

# Ensure analysis modules can be imported
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(ANALYSIS_DIR)
sys.path.insert(0, ANALYSIS_DIR)

from phase4_features import (
    normalize_business_name,
    first_token,
    extract_domain_stem,
    extract_digit_blocks,
    tokenize,
    char_ngrams,
    token_jaccard,
    token_overlap,
    build_record_profile,
    extract_feature_vector,
)
from phase4_candidate_generation import first_2_tokens

# Dataset Paths
TEST_S1_PATH = os.path.join(WORKSPACE_ROOT, "dataset", "test", "test_source1.tsv")
MODEL_PATH = os.path.join(ANALYSIS_DIR, "phase4_model.joblib")

# Baseline comparison artifacts
BASELINE_CANDIDATE_PATH = os.path.join(ANALYSIS_DIR, "benchmark_1k_candidate_pairs.tsv")
BASELINE_MATCHING_PATH = os.path.join(ANALYSIS_DIR, "benchmark_1k_matching_results.tsv")

# Output benchmark artifacts
OUTPUT_CANDIDATE_PATH = os.path.join(ANALYSIS_DIR, "parallel_benchmark_1k_candidate_pairs.tsv")
OUTPUT_MATCHING_PATH = os.path.join(ANALYSIS_DIR, "parallel_benchmark_1k_matching_results.tsv")

# Reusable index storage paths
SCRATCH_DIR = os.path.join(WORKSPACE_ROOT, "scratch")
S2_INDEX_DB_PATH = os.path.join(SCRATCH_DIR, "s2_index.db")
S3_INDEX_DB_PATH = os.path.join(SCRATCH_DIR, "s3_index.db")

# Benchmark Configuration
BENCHMARK_SAMPLE_SIZE = 1000
MAX_CANDIDATES_PER_SOURCE = 30


def load_test_s1_slice(file_path: str, sample_size: int = 1000) -> List[Dict[str, str]]:
    """Load the first N Source 1 test records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()  # Skip header
        for line in f:
            if len(records) >= sample_size:
                break
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 4:
                records.append({
                    "entity_id": parts[0].strip(),
                    "name": parts[1],
                    "addr": parts[2],
                    "country": parts[3].strip()
                })
    return records


def query_source_index(
    conn: sqlite3.Connection,
    s1_prof: Dict[str, Any],
    max_cands: int = MAX_CANDIDATES_PER_SOURCE
) -> List[Tuple[str, str, str]]:
    """
    Query source index for a single S1 profile.
    Returns up to max_cands (eid, bname, baddr) tuples,
    selected as the first matching rows in line order (ascending row_id).
    """
    country = s1_prof["country"]
    s1_norm = s1_prof["norm_name"]
    s1_f2 = first_2_tokens(s1_norm) if s1_norm else ""
    s1_stripped = s1_prof["stripped_name"]
    s1_f1 = s1_prof["first_tok"]
    s1_digs = s1_prof["digits"]
    s1_tokens = s1_prof["name_tokens"]
    s1_ngrams = s1_prof["name_ngrams"]

    cand_store: Dict[int, Tuple[str, str, str]] = {}
    cur = conn.cursor()

    # Pass 1: Exact Name Match
    if s1_norm:
        cur.execute(
            "SELECT row_id, eid, bname, baddr FROM records WHERE norm_name = ? AND country = ?",
            (s1_norm, country)
        )
        for row in cur.fetchall():
            rid, eid, bname, baddr = row
            if rid not in cand_store:
                cand_store[rid] = (eid, bname, baddr)

    # Pass 2: Domain Stem Match
    if len(s1_stripped) >= 4:
        cur.execute(
            "SELECT row_id, eid, bname, baddr FROM records WHERE dom = ? AND country = ?",
            (s1_stripped, country)
        )
        for row in cur.fetchall():
            rid, eid, bname, baddr = row
            if rid not in cand_store:
                cand_store[rid] = (eid, bname, baddr)

    # Pass 3: First-2-Tokens Prefix Match (Fuzzy Name)
    if s1_f2 and len(s1_f2) >= 4:
        cur.execute(
            "SELECT row_id, eid, bname, baddr, norm_name FROM records WHERE f2 = ? AND country = ?",
            (s1_f2, country)
        )
        for row in cur.fetchall():
            rid, eid, bname, baddr, cand_norm = row
            if rid not in cand_store:
                t_toks = tokenize(bname)
                t_ngs = char_ngrams(cand_norm, 3)
                tj = token_jaccard(s1_tokens, t_toks)
                gj = token_jaccard(s1_ngrams, t_ngs)
                if tj >= 0.40 or gj >= 0.45:
                    cand_store[rid] = (eid, bname, baddr)

    # Pass 4: Location-Anchored Digit Block + First Token Match
    if len(s1_f1) >= 4 and s1_digs:
        for d in s1_digs:
            cur.execute(
                "SELECT d.row_id, r.eid, r.bname, r.baddr "
                "FROM digits d JOIN records r ON d.row_id = r.row_id "
                "WHERE d.d = ? AND d.f1 = ? AND d.country = ?",
                (d, s1_f1, country)
            )
            for row in cur.fetchall():
                rid, eid, bname, baddr = row
                if rid not in cand_store:
                    t_toks = tokenize(bname)
                    if token_overlap(s1_tokens, t_toks) >= 1:
                        cand_store[rid] = (eid, bname, baddr)

    if not cand_store:
        return []

    # Capping: select up to max_cands by ascending row_id (exact chronological file order)
    sorted_rids = sorted(cand_store.keys())[:max_cands]
    return [cand_store[rid] for rid in sorted_rids]


def worker_task(country: str, s1_slice: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Worker task: Processes S1 entities for a specific country.
    """
    t_start = time.perf_counter()
    
    # 1. Load model independently in this process
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    tau = artifact["selected_tau"]
    tau_singleton = artifact["selected_tau_singleton"]
    
    # 2. Open read-only SQLite connections
    s2_uri = f"file:{S2_INDEX_DB_PATH}?mode=ro"
    s3_uri = f"file:{S3_INDEX_DB_PATH}?mode=ro"
    conn_s2 = sqlite3.connect(s2_uri, uri=True)
    conn_s3 = sqlite3.connect(s3_uri, uri=True)
    
    # Optional pragmas for performance
    conn_s2.execute("PRAGMA cache_size = -131072;")
    conn_s3.execute("PRAGMA cache_size = -131072;")

    # Process S1 entities
    s1_profiles = {}
    for rec in s1_slice:
        eid = rec["entity_id"]
        s1_profiles[eid] = build_record_profile(rec["name"], rec["addr"], rec["country"])
        
    all_candidate_records = {}
    ordered_candidate_ids = {}
    matching = {}
    
    for rec in s1_slice:
        s1_id = rec["entity_id"]
        s1_prof = s1_profiles[s1_id]
        
        # Query S2 and S3
        s2_matches = query_source_index(conn_s2, s1_prof, max_cands=MAX_CANDIDATES_PER_SOURCE)
        s3_matches = query_source_index(conn_s3, s1_prof, max_cands=MAX_CANDIDATES_PER_SOURCE)
        
        cand_dict: Dict[str, Tuple[str, str, bool]] = {}
        s2_cids = []
        for eid, bname, baddr in s2_matches:
            cand_dict[eid] = (bname, baddr, False)
            s2_cids.append(eid)
            
        s3_cids = []
        for eid, bname, baddr in s3_matches:
            cand_dict[eid] = (bname, baddr, True)
            s3_cids.append(eid)
            
        s2_cids.sort()
        s3_cids.sort()
        cids_ordered = s2_cids + s3_cids
        
        all_candidate_records[s1_id] = cand_dict
        ordered_candidate_ids[s1_id] = cids_ordered
        
        # Scoring
        if not cids_ordered:
            matching[s1_id] = []
            continue
            
        X = []
        for cid in cids_ordered:
            bname, baddr, is_s3 = cand_dict[cid]
            cand_prof = build_record_profile(bname, baddr, country)
            X.append(extract_feature_vector(s1_prof, cand_prof, is_s3, len(cids_ordered)))
            
        probs = model.predict_proba(np.array(X, dtype=np.float32))[:, 1]
        max_p = float(np.max(probs))
        
        if max_p < tau_singleton:
            matching[s1_id] = []
        else:
            matching[s1_id] = [c for c, p in zip(cids_ordered, probs) if p >= tau]
            
    conn_s2.close()
    conn_s3.close()
    
    t_elapsed = time.perf_counter() - t_start
    return {
        "country": country,
        "elapsed": t_elapsed,
        "ordered_candidates": ordered_candidate_ids,
        "matching": matching
    }


def compare_files_exact(file_opt: str, file_base: str, label: str) -> List[str]:
    """Verify exact equivalence line by line."""
    mismatches = []
    if not os.path.isfile(file_opt):
        return [f"File not found: {file_opt}"]
    if not os.path.isfile(file_base):
        return [f"File not found: {file_base}"]

    with open(file_opt, 'r', encoding='utf-8') as f_opt, open(file_base, 'r', encoding='utf-8') as f_base:
        lines_opt = f_opt.readlines()
        lines_base = f_base.readlines()

    if len(lines_opt) != len(lines_base):
        mismatches.append(f"{label}: Line count mismatch ({len(lines_opt)} vs {len(lines_base)})")
        return mismatches

    for idx, (l_opt, l_base) in enumerate(zip(lines_opt, lines_base), 1):
        if l_opt != l_base:
            mismatches.append(f"{label} line {idx} mismatch:\n  Parallel : {l_opt.rstrip()!r}\n  Baseline: {l_base.rstrip()!r}")
            if len(mismatches) >= 5:
                break

    return mismatches


def compute_sha256(filepath: str) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def main():
    print("=" * 80)
    print("Starting Amazon ML Challenge 2026 — Phase 4 Step 3A: Parallel Benchmark")
    print("=" * 80)

    tracemalloc.start()
    t_global_start = time.perf_counter()

    # Step 1: Load and profile the first 1,000 Source 1 test records
    t_s1_start = time.perf_counter()
    print(f"\nLoading first {BENCHMARK_SAMPLE_SIZE:,} Source 1 records from {os.path.relpath(TEST_S1_PATH)}...")
    s1_slice = load_test_s1_slice(TEST_S1_PATH, sample_size=BENCHMARK_SAMPLE_SIZE)
    s1_order = [rec["entity_id"] for rec in s1_slice]
    print(f"Loaded {len(s1_slice):,} records.")
    
    # Partition by country
    country_partitions = collections.defaultdict(list)
    for rec in s1_slice:
        country_partitions[rec["country"]].append(rec)
        
    s1_index_time = time.perf_counter() - t_s1_start
    print(f"Source 1 profiling completed in {s1_index_time:.3f} s.")
    print("Partitions:")
    for country, items in country_partitions.items():
        print(f"  {country}: {len(items)} entities")

    # Step 2: Query the indexes and process S1 entities using multiprocessing
    print(f"\nProcessing {len(s1_order):,} Source 1 entities across {len(country_partitions)} workers...")
    t_query_start = time.perf_counter()

    final_candidates = {}
    final_matching = {}
    worker_times = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(country_partitions)) as executor:
        futures = {
            executor.submit(worker_task, country, items): country
            for country, items in country_partitions.items()
        }
        
        for future in concurrent.futures.as_completed(futures):
            country = futures[future]
            try:
                res = future.result()
                final_candidates.update(res["ordered_candidates"])
                final_matching.update(res["matching"])
                worker_times[country] = res["elapsed"]
                print(f"  Worker [{country}] finished in {res['elapsed']:.2f} s")
            except Exception as e:
                print(f"Worker [{country}] failed with exception: {e}")
                sys.exit(1)

    processing_time = time.perf_counter() - t_query_start
    total_candidates = sum(len(c) for c in final_candidates.values())
    total_matches = sum(len(m) for m in final_matching.values())
    print(f"Parallel processing completed in {processing_time:.2f} s. Retrieved {total_candidates:,} candidates.")

    # Step 3: Write outputs in original S1 order
    print("\nWriting output files...")
    t_write_start = time.perf_counter()
    with open(OUTPUT_MATCHING_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            cids_str = ",".join(final_matching.get(eid, []))
            f.write(f"{eid}\t{cids_str}\n")

    with open(OUTPUT_CANDIDATE_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            cids_str = ",".join(final_candidates.get(eid, []))
            f.write(f"{eid}\t{cids_str}\n")
    write_time = time.perf_counter() - t_write_start

    # Stop memory tracking
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    total_time = time.perf_counter() - t_global_start

    print("\n" + "=" * 80)
    print("STRICT VERIFICATION AGAINST BASELINE")
    print("=" * 80)
    
    cand_mismatches = compare_files_exact(OUTPUT_CANDIDATE_PATH, BASELINE_CANDIDATE_PATH, "Candidates")
    match_mismatches = compare_files_exact(OUTPUT_MATCHING_PATH, BASELINE_MATCHING_PATH, "Matches")
    
    if cand_mismatches or match_mismatches:
        print("CRITICAL ERROR: Parallel output differs from baseline!", file=sys.stderr)
        for err in cand_mismatches + match_mismatches:
            print(f"  [MISMATCH] {err}", file=sys.stderr)
        sys.exit(1)
        
    sha_cand_base = compute_sha256(BASELINE_CANDIDATE_PATH)
    sha_cand_para = compute_sha256(OUTPUT_CANDIDATE_PATH)
    sha_match_base = compute_sha256(BASELINE_MATCHING_PATH)
    sha_match_para = compute_sha256(OUTPUT_MATCHING_PATH)
    
    cand_match = (sha_cand_base == sha_cand_para)
    match_match = (sha_match_base == sha_match_para)
    
    if not (cand_match and match_match):
        print("CRITICAL ERROR: SHA-256 mismatch against baseline!", file=sys.stderr)
        print(f"  Candidate Baseline: {sha_cand_base}")
        print(f"  Candidate Parallel: {sha_cand_para}")
        print(f"  Matching Baseline:  {sha_match_base}")
        print(f"  Matching Parallel:  {sha_match_para}")
        sys.exit(1)
        
    print("SUCCESS: 100% BYTE-FOR-BYTE IDENTICAL OUTPUT TO BASELINE!")
    print(f"  Candidate SHA-256 : {sha_cand_para} (match={cand_match})")
    print(f"  Matching  SHA-256 : {sha_match_para} (match={match_match})")
    print(f"  S1 entities       : {len(s1_order):,}")
    print(f"  Candidates        : {total_candidates:,}")
    print(f"  Accepted matches  : {total_matches:,}")

    print("\n" + "=" * 80)
    print("PARALLEL BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Component':<42} | {'Value'}")
    print("-" * 80)
    print(f"{'S1 Profiling Time':<42} | {s1_index_time:.3f} s")
    print(f"{'Parallel Processing Time':<42} | {processing_time:.2f} s")
    for c, t in worker_times.items():
        print(f"{f'  Worker [{c}] Execution Time':<42} | {t:.2f} s")
    print(f"{'Merge & Write Time':<42} | {write_time:.2f} s")
    print(f"{'Total End-to-End Runtime':<42} | {total_time:.2f} s")
    print(f"{'Peak Process RAM (Parent Only)':<42} | {peak_mem / (1024*1024):.2f} MB")
    print(f"{'Total Candidate Count':<42} | {total_candidates:,}")
    print(f"{'Total Accepted Matches':<42} | {total_matches:,}")
    print("=" * 80)
    
    # Compare with indexed baseline
    baseline_time = 713.68 # From previous measurement
    speedup = baseline_time / total_time
    print("\n" + "=" * 80)
    print("COMPARISON VS INDEXED BASELINE")
    print("=" * 80)
    print(f"{'Metric':<42} | {'Indexed':<14} | {'Parallel':<14} | {'Speedup'}")
    print("-" * 80)
    print(f"{'Total Runtime':<42} | {baseline_time:<14.2f} | {total_time:<14.2f} | {speedup:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
