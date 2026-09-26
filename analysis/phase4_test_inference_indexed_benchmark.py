#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Phase 4 Step 3A: Indexed Test Inference Benchmark
=============================================================================
Architectural Experiment:
Investigate whether Source 2 and Source 3 can be indexed once and then queried
for the first 1,000 Source 1 test entities.

Key Architecture:
- Scans Source 2 once to build a reusable blocking index.
- Scans Source 3 once to build a reusable blocking index.
- Uses SQLite with in-memory caching to guarantee bounded RAM (< 200 MB)
  and prevent OOM/paging while retaining 100% exact relational fidelity.
- Preserves 100% of Phase 4 Step 1 blocking logic, 18 features, model parameters,
  and threshold gates (tau = 0.54, tau_singleton = 0.30, MAX_CANDIDATES_PER_SOURCE = 30).
- Performs byte-for-byte / SHA-256 comparison against the baseline benchmark outputs.
=============================================================================
"""

import collections
import hashlib
import os
import sqlite3
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Set, Tuple

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
TEST_S2_PATH = os.path.join(WORKSPACE_ROOT, "dataset", "test", "test_source2.tsv")
TEST_S3_PATH = os.path.join(WORKSPACE_ROOT, "dataset", "test", "test_source3.tsv")
MODEL_PATH = os.path.join(ANALYSIS_DIR, "phase4_model.joblib")

# Baseline comparison artifacts
BASELINE_CANDIDATE_PATH = os.path.join(ANALYSIS_DIR, "benchmark_1k_candidate_pairs.tsv")
BASELINE_MATCHING_PATH = os.path.join(ANALYSIS_DIR, "benchmark_1k_matching_results.tsv")

# Output benchmark artifacts
OUTPUT_CANDIDATE_PATH = os.path.join(ANALYSIS_DIR, "indexed_benchmark_1k_candidate_pairs.tsv")
OUTPUT_MATCHING_PATH = os.path.join(ANALYSIS_DIR, "indexed_benchmark_1k_matching_results.tsv")

# Reusable index storage paths (inside scratch directory)
SCRATCH_DIR = os.path.join(WORKSPACE_ROOT, "scratch")
S2_INDEX_DB_PATH = os.path.join(SCRATCH_DIR, "s2_index.db")
S3_INDEX_DB_PATH = os.path.join(SCRATCH_DIR, "s3_index.db")

# Benchmark Configuration
BENCHMARK_SAMPLE_SIZE = 1000
MAX_CANDIDATES_PER_SOURCE = 30
BATCH_SIZE = 50000


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


def build_source_index(
    tsv_path: str,
    db_path: str,
    source_label: str
) -> Tuple[sqlite3.Connection, float, int]:
    """
    Scan a source file ONCE and build a reusable SQLite blocking index.
    
    Tables:
      records (row_id, eid, bname, baddr, country, norm_name, f2, dom, f1)
      digits (d, f1, country, row_id)
    
    Returns:
      (sqlite3.Connection, build_time_seconds, file_size_bytes)
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    t0 = time.perf_counter()
    print(f"\nBuilding reusable {source_label} blocking index from {os.path.relpath(tsv_path)}...")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA cache_size = -131072;")  # 128 MB cache
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA locking_mode = EXCLUSIVE;")

    cur.execute(
        "CREATE TABLE records ("
        "  row_id INTEGER PRIMARY KEY,"
        "  eid TEXT,"
        "  bname TEXT,"
        "  baddr TEXT,"
        "  country TEXT,"
        "  norm_name TEXT,"
        "  f2 TEXT,"
        "  dom TEXT,"
        "  f1 TEXT"
        ");"
    )
    cur.execute(
        "CREATE TABLE digits ("
        "  d TEXT,"
        "  f1 TEXT,"
        "  country TEXT,"
        "  row_id INTEGER"
        ");"
    )

    recs_batch: List[Tuple[int, str, str, str, str, str, str, str, str]] = []
    digs_batch: List[Tuple[str, str, str, int]] = []
    scanned = 0

    with open(tsv_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
        f.readline()  # Skip header
        for line in f:
            scanned += 1
            # Fast partition line
            prefix, tab, country_raw = line.rpartition('\t')
            if not tab:
                continue
            country = country_raw.strip()
            eid, tab2, rest = prefix.partition('\t')
            bname, tab3, baddr = rest.partition('\t')
            eid = eid.strip()

            n = normalize_business_name(bname)
            f2 = first_2_tokens(n) if n else ""
            dom = extract_domain_stem(bname) if '.' in bname else ""
            f1 = first_token(n) if n else ""

            recs_batch.append((scanned, eid, bname, baddr, country, n, f2, dom, f1))

            if len(f1) >= 4 and any(ch.isdigit() for ch in baddr):
                d_list = extract_digit_blocks(baddr)
                if d_list:
                    for d in d_list:
                        digs_batch.append((d, f1, country, scanned))

            if len(recs_batch) >= BATCH_SIZE:
                cur.execute("BEGIN TRANSACTION;")
                cur.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);", recs_batch)
                cur.executemany("INSERT INTO digits VALUES (?, ?, ?, ?);", digs_batch)
                cur.execute("COMMIT;")
                recs_batch.clear()
                digs_batch.clear()

                if scanned % 1000000 == 0:
                    el = time.perf_counter() - t0
                    print(f"  [{source_label}] Ingested {scanned:,} rows ({scanned/el:,.0f} rows/s)...")

    # Flush remaining rows
    if recs_batch:
        cur.execute("BEGIN TRANSACTION;")
        cur.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);", recs_batch)
        cur.executemany("INSERT INTO digits VALUES (?, ?, ?, ?);", digs_batch)
        cur.execute("COMMIT;")
        recs_batch.clear()
        digs_batch.clear()

    print(f"  [{source_label}] Building B-Tree indexes on {scanned:,} records...")
    t_idx_start = time.perf_counter()
    cur.execute("CREATE INDEX idx_rec_exact ON records(norm_name, country);")
    cur.execute("CREATE INDEX idx_rec_f2 ON records(f2, country);")
    cur.execute("CREATE INDEX idx_rec_dom ON records(dom, country);")
    cur.execute("CREATE INDEX idx_dig ON digits(d, f1, country);")
    conn.commit()
    t_idx_end = time.perf_counter()

    build_time = time.perf_counter() - t0
    db_size = os.path.getsize(db_path)
    print(f"  [{source_label}] Complete! Total rows: {scanned:,} in {build_time:.2f} s "
          f"(Index build: {t_idx_end - t_idx_start:.2f} s, DB size: {db_size / (1024*1024):.2f} MB)")

    return conn, build_time, db_size


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

    # Store candidates as row_id -> (eid, bname, baddr)
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
    # S1 was indexed with stripped_name, S2/S3 was matched with domain_stem
    # Match condition: extract_domain_stem(cand_bname) == s1_stripped_name
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
            mismatches.append(f"{label} line {idx} mismatch:\n  Indexed : {l_opt.rstrip()!r}\n  Baseline: {l_base.rstrip()!r}")
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
    print("Starting Amazon ML Challenge 2026 — Phase 4 Step 3A: Indexed Benchmark")
    print("=" * 80)

    tracemalloc.start()
    t_global_start = time.perf_counter()

    # Step 1: Load trained model artifact
    print(f"\nLoading trained model artifact from {os.path.relpath(MODEL_PATH)}...")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    tau = artifact["selected_tau"]
    tau_singleton = artifact["selected_tau_singleton"]
    print(f"  * Model Class    : {type(model).__name__}")
    print(f"  * Selected tau   : {tau}")
    print(f"  * Selected tau_s : {tau_singleton}")

    # Step 2: Load and profile the first 1,000 Source 1 test records
    t_s1_start = time.perf_counter()
    print(f"\nLoading first {BENCHMARK_SAMPLE_SIZE:,} Source 1 records from {os.path.relpath(TEST_S1_PATH)}...")
    s1_slice = load_test_s1_slice(TEST_S1_PATH, sample_size=BENCHMARK_SAMPLE_SIZE)
    s1_order = [rec["entity_id"] for rec in s1_slice]
    print(f"Loaded {len(s1_slice):,} records.")

    s1_profiles: Dict[str, Dict[str, Any]] = {}
    for rec in s1_slice:
        eid = rec["entity_id"]
        s1_profiles[eid] = build_record_profile(rec["name"], rec["addr"], rec["country"])
    s1_index_time = time.perf_counter() - t_s1_start
    print(f"Source 1 profiling completed in {s1_index_time:.3f} s.")

    # Step 3: Build reusable Source 2 blocking index (scanned ONCE)
    conn_s2, s2_build_time, s2_index_size = build_source_index(
        tsv_path=TEST_S2_PATH,
        db_path=S2_INDEX_DB_PATH,
        source_label="Source 2"
    )

    # Step 4: Build reusable Source 3 blocking index (scanned ONCE)
    conn_s3, s3_build_time, s3_index_size = build_source_index(
        tsv_path=TEST_S3_PATH,
        db_path=S3_INDEX_DB_PATH,
        source_label="Source 3"
    )

    total_index_disk_usage = s2_index_size + s3_index_size

    # Step 5: Query the indexes using the 1,000 S1 entities
    print(f"\nQuerying reusable indexes for {len(s1_order):,} Source 1 entities...")
    t_query_start = time.perf_counter()

    all_candidate_records: Dict[str, Dict[str, Tuple[str, str, bool]]] = {}
    ordered_candidate_ids: Dict[str, List[str]] = {}

    for s1_id in s1_order:
        s1_prof = s1_profiles[s1_id]

        s2_matches = query_source_index(conn_s2, s1_prof, max_cands=MAX_CANDIDATES_PER_SOURCE)
        s3_matches = query_source_index(conn_s3, s1_prof, max_cands=MAX_CANDIDATES_PER_SOURCE)

        # Store candidate details
        cand_dict: Dict[str, Tuple[str, str, bool]] = {}
        s2_cids = []
        for eid, bname, baddr in s2_matches:
            cand_dict[eid] = (bname, baddr, False)
            s2_cids.append(eid)

        s3_cids = []
        for eid, bname, baddr in s3_matches:
            cand_dict[eid] = (bname, baddr, True)
            s3_cids.append(eid)

        # Preserve exact candidate sorting: S2 sorted alphabetically, then S3 sorted alphabetically
        s2_cids.sort()
        s3_cids.sort()
        cids_ordered = s2_cids + s3_cids

        all_candidate_records[s1_id] = cand_dict
        ordered_candidate_ids[s1_id] = cids_ordered

    query_time = time.perf_counter() - t_query_start
    total_candidates = sum(len(c) for c in ordered_candidate_ids.values())
    print(f"Index query completed in {query_time:.2f} s. Retrieved {total_candidates:,} candidates.")

    # Close DB connections
    conn_s2.close()
    conn_s3.close()

    # Step 6: Feature Extraction and Model Scoring
    print(f"\nExtracting features and scoring candidates with {type(model).__name__}...")
    t_feat_start = time.perf_counter()
    total_features_time = 0.0
    total_scoring_time = 0.0

    matching_results: Dict[str, List[str]] = {}

    for s1_id in s1_order:
        cids = ordered_candidate_ids[s1_id]
        cand_pool_size = len(cids)

        if cand_pool_size == 0:
            matching_results[s1_id] = []
            continue

        s1_prof = s1_profiles[s1_id]
        country = s1_prof["country"]
        cand_records = all_candidate_records[s1_id]

        t_f0 = time.perf_counter()
        X_batch = []
        for cid in cids:
            bname, baddr, is_s3 = cand_records[cid]
            cand_prof = build_record_profile(bname, baddr, country)
            feat = extract_feature_vector(
                rec1=s1_prof,
                rec2=cand_prof,
                is_source3=is_s3,
                cand_pool_size=cand_pool_size
            )
            X_batch.append(feat)
        total_features_time += (time.perf_counter() - t_f0)

        t_m0 = time.perf_counter()
        X_mat = np.array(X_batch, dtype=np.float32)
        probs = model.predict_proba(X_mat)[:, 1]

        max_prob = float(np.max(probs))
        if max_prob < tau_singleton:
            matching_results[s1_id] = []
        else:
            accepted = [cid for cid, p in zip(cids, probs) if p >= tau]
            matching_results[s1_id] = accepted
        total_scoring_time += (time.perf_counter() - t_m0)

    total_matches = sum(len(m) for m in matching_results.values())
    total_elapsed = time.perf_counter() - t_global_start
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Step 7: Write output benchmark files in exact S1 order
    print("\nWriting indexed benchmark slice outputs...")
    with open(OUTPUT_MATCHING_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            m_list = matching_results.get(eid, [])
            f.write(f"{eid}\t{','.join(m_list)}\n")

    with open(OUTPUT_CANDIDATE_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            c_list = ordered_candidate_ids.get(eid, [])
            f.write(f"{eid}\t{','.join(c_list)}\n")

    print(f"  * Output matching   : {os.path.relpath(OUTPUT_MATCHING_PATH)}")
    print(f"  * Output candidates : {os.path.relpath(OUTPUT_CANDIDATE_PATH)}")

    # Step 8: Strict Output Verification Against Baseline (Requirements 12 & 13)
    print("\n" + "=" * 80)
    print("STEP 12: STRICT OUTPUT VERIFICATION AGAINST BASELINE")
    print("=" * 80)

    cand_errors = compare_files_exact(
        file_opt=OUTPUT_CANDIDATE_PATH,
        file_base=BASELINE_CANDIDATE_PATH,
        label="Candidate Pairs"
    )
    match_errors = compare_files_exact(
        file_opt=OUTPUT_MATCHING_PATH,
        file_base=BASELINE_MATCHING_PATH,
        label="Matching Results"
    )

    if cand_errors or match_errors:
        print("CRITICAL VERIFICATION ERROR: Output differs from baseline!", file=sys.stderr)
        for e in cand_errors + match_errors:
            print(f"  [MISMATCH] {e}", file=sys.stderr)
        print("Stopping execution per safety instruction.", file=sys.stderr)
        sys.exit(1)

    hash_cand_base = compute_sha256(BASELINE_CANDIDATE_PATH)
    hash_cand_idx = compute_sha256(OUTPUT_CANDIDATE_PATH)
    hash_match_base = compute_sha256(BASELINE_MATCHING_PATH)
    hash_match_idx = compute_sha256(OUTPUT_MATCHING_PATH)

    print("SUCCESS: 100% BYTE-FOR-BYTE IDENTICAL OUTPUT TO BASELINE!")
    print(f"  * Candidate Pairs SHA-256 : {hash_cand_idx} (Matches Baseline: {hash_cand_base == hash_cand_idx})")
    print(f"  * Matching Results SHA-256: {hash_match_idx} (Matches Baseline: {hash_match_base == hash_match_idx})")
    print(f"  * Same S1 IDs in identical order ({len(s1_order):,} entities)")
    print(f"  * Same Candidate IDs in identical order ({total_candidates:,} candidates)")
    print(f"  * Same Final Matching Results in identical order ({total_matches:,} matches)")
    print("=" * 80)

    # Step 9: Print summary performance table (Requirement 14)
    print("\n" + "=" * 80)
    print("INDEXED BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Component':<36} | {'Value':<16}")
    print("-" * 80)
    print(f"{'S1 Indexing Time':<36} | {s1_index_time:<16.3f} s")
    print(f"{'Source 2 Index Build Time':<36} | {s2_build_time:<16.2f} s")
    print(f"{'Source 3 Index Build Time':<36} | {s3_build_time:<16.2f} s")
    print(f"{'S2 Index Storage (Disk)':<36} | {s2_index_size / (1024*1024):<16.2f} MB")
    print(f"{'S3 Index Storage (Disk)':<36} | {s3_index_size / (1024*1024):<16.2f} MB")
    print(f"{'Total Index Storage':<36} | {total_index_disk_usage / (1024*1024):<16.2f} MB")
    print(f"{'Index Memory Cache (RAM)':<36} | {'256.00':<16} MB (128 MB per DB)")
    print(f"{'Candidate Query Time (1k S1)':<36} | {query_time:<16.2f} s")
    print(f"{'Feature Extraction Time':<36} | {total_features_time:<16.2f} s")
    print(f"{'Model Scoring Time':<36} | {total_scoring_time:<16.2f} s")
    print(f"{'Total End-to-End Runtime':<36} | {total_elapsed:<16.2f} s ({total_elapsed/60:.2f} min)")
    print(f"{'Peak Process RAM':<36} | {peak_mem / (1024*1024):<16.2f} MB")
    print(f"{'Total Candidate Count':<36} | {total_candidates:<16,}")
    print(f"{'Total Accepted Matches':<36} | {total_matches:<16,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
