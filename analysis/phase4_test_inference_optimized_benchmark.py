#!/usr/bin/env python3
"""
Phase 4 Step 3A (Optimized): High-Throughput Memory-Safe Test Inference Benchmark
Amazon ML Challenge 2026 — Business Entity Resolution

This module implements an optimized test inference pipeline for the representative
slice of 1,000 Source 1 test entities, mathematically identical to Phase 4 Step 1:
- Country-partitioned scanning: S2 and S3 rows for other countries are skipped immediately.
- Fast country pre-check via rpartition('\t') on raw TSV lines.
- Digit pre-check: only inspects/extracts digit blocks when (f1 in dig_index) and address has digits.
- Fast domain check: only inspects domain stem when '.' is present in business name.
- Exact candidate generation logic, 18 features, model parameters (tau=0.54, tau_singleton=0.30),
  and MAX_CANDIDATES_PER_SOURCE = 30 are strictly preserved.
- Verifies exact 1:1 match against baseline benchmark_1k_candidate_pairs.tsv and
  benchmark_1k_matching_results.tsv.
"""

import os
import sys
import time
import collections
import tracemalloc
from typing import Dict, Set, List, Tuple, Any, Optional

import numpy as np
import joblib

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
from phase4_candidate_generation import first_2_tokens

# File paths
DATASET_TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
TEST_S1_PATH = os.path.join(DATASET_TEST_DIR, "test_source1.tsv")
TEST_S2_PATH = os.path.join(DATASET_TEST_DIR, "test_source2.tsv")
TEST_S3_PATH = os.path.join(DATASET_TEST_DIR, "test_source3.tsv")

MODEL_PATH = os.path.join(CURRENT_DIR, "phase4_model.joblib")

# Existing baseline files to compare against
BASELINE_MATCHING_PATH = os.path.join(CURRENT_DIR, "benchmark_1k_matching_results.tsv")
BASELINE_CANDIDATE_PATH = os.path.join(CURRENT_DIR, "benchmark_1k_candidate_pairs.tsv")

# New optimized output files
OPTIMIZED_MATCHING_PATH = os.path.join(CURRENT_DIR, "optimized_benchmark_1k_matching_results.tsv")
OPTIMIZED_CANDIDATE_PATH = os.path.join(CURRENT_DIR, "optimized_benchmark_1k_candidate_pairs.tsv")

BENCHMARK_SAMPLE_SIZE = 1000
MAX_CANDIDATES_PER_SOURCE = 30


def load_test_s1_slice(
    s1_path: str = TEST_S1_PATH,
    sample_size: int = BENCHMARK_SAMPLE_SIZE
) -> List[Dict[str, str]]:
    """Load the first sample_size Source 1 test records in original order."""
    records = []
    with open(s1_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 4:
                records.append({
                    "entity_id": parts[0].strip(),
                    "name": parts[1],
                    "addr": parts[2],
                    "country": parts[3].strip()
                })
            if len(records) >= sample_size:
                break
    return records


class CountryPartitionInferenceEngine:
    """
    Optimized country-partitioned inference engine.
    Processes one country at a time, skipping non-matching rows in microseconds.
    """

    def __init__(self, model_artifact_path: str = MODEL_PATH):
        if not os.path.isfile(model_artifact_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_artifact_path}")

        print(f"Loading trained model artifact from {os.path.relpath(model_artifact_path)}...")
        artifact = joblib.load(model_artifact_path)
        self.model = artifact["model"]
        self.tau = float(artifact.get("selected_tau", 0.54))
        self.tau_singleton = float(artifact.get("selected_tau_singleton", 0.30))
        self.feature_names = artifact.get("feature_names", FEATURE_NAMES)

        print(f"  * Model Class    : {self.model.__class__.__name__}")
        print(f"  * Selected tau   : {self.tau:.2f}")
        print(f"  * Selected tau_s : {self.tau_singleton:.2f}")

    def run_country_partition(
        self,
        country: str,
        records: List[Dict[str, str]],
        s2_path: str = TEST_S2_PATH,
        s3_path: str = TEST_S3_PATH
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, float]]:
        """
        Runs candidate generation, feature extraction, and scoring for one country partition.
        """
        # Step 1: Index S1 records for this country
        t_idx_start = time.perf_counter()
        s1_profiles: Dict[str, Dict[str, Any]] = {}
        s1_order: List[str] = []

        block_exact: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        block_f2: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        block_dom: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        block_dig: Dict[Tuple[str, str, str], List[str]] = collections.defaultdict(list)
        f1_in_dig: Set[str] = set()

        for rec in records:
            eid = rec["entity_id"]
            s1_order.append(eid)
            p = build_record_profile(rec["name"], rec["addr"], country)
            s1_profiles[eid] = p

            n = p["norm_name"]
            stem = p["stripped_name"]
            f1 = p["first_tok"]
            digs = p["digits"]

            # 1. Exact Name Index
            if n:
                block_exact[(n, country)].append(eid)

            # 2. First-2-Tokens Prefix Index
            f2 = first_2_tokens(n)
            if len(f2) >= 4:
                block_f2[(f2, country)].append(eid)

            # 3. Domain Stem Index
            if len(stem) >= 4:
                block_dom[(stem, country)].append(eid)

            # 4. Street/Postal Digit + First Token Index
            if len(f1) >= 4 and digs:
                f1_in_dig.add(f1)
                for d in digs:
                    block_dig[(d, f1, country)].append(eid)

        idx_time = time.perf_counter() - t_idx_start

        # Candidate store: s1_id -> {cand_id: (bname, baddr, is_s3)}
        cand_store: Dict[str, Dict[str, Tuple[str, str, bool]]] = {
            eid: {} for eid in s1_order
        }
        s2_counts: Dict[str, int] = collections.defaultdict(int)
        s3_counts: Dict[str, int] = collections.defaultdict(int)

        # Step 2: Stream S2 (filtered for this country)
        t_s2_start = time.perf_counter()
        s2_hits = 0
        s2_scanned = 0
        with open(s2_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
            f.readline()  # Skip header
            for line in f:
                s2_scanned += 1
                # Optimization B & C: Fast country check on raw line via rpartition
                prefix, tab, country_raw = line.rpartition('\t')
                if not tab or country_raw.strip() != country:
                    continue

                eid, tab2, rest = prefix.partition('\t')
                bname, tab3, baddr = rest.partition('\t')
                eid = eid.strip()

                n = normalize_business_name(bname)

                # Pass 1: Exact Name
                k_ex = (n, country) if n else None
                if k_ex and k_ex in block_exact:
                    for s1_id in block_exact[k_ex]:
                        if s2_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE:
                            if eid not in cand_store[s1_id]:
                                cand_store[s1_id][eid] = (bname, baddr, False)
                                s2_counts[s1_id] += 1
                                s2_hits += 1

                # Pass 2: Domain Stem (only if '.' is in bname)
                if '.' in bname:
                    dom_s = extract_domain_stem(bname)
                    if dom_s and len(dom_s) >= 4:
                        k_dom = (dom_s, country)
                        if k_dom in block_dom:
                            for s1_id in block_dom[k_dom]:
                                if s2_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE:
                                    if eid not in cand_store[s1_id]:
                                        cand_store[s1_id][eid] = (bname, baddr, False)
                                        s2_counts[s1_id] += 1
                                        s2_hits += 1

                # Pass 3: First-2-Tokens Prefix Match
                f2 = first_2_tokens(n)
                if len(f2) >= 4:
                    k_f2 = (f2, country)
                    if k_f2 in block_f2:
                        t_toks = None
                        t_ngs = None
                        for s1_id in block_f2[k_f2]:
                            if s2_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE and eid not in cand_store[s1_id]:
                                if t_toks is None:
                                    t_toks = tokenize(bname)
                                    t_ngs = char_ngrams(n, 3)
                                s1 = s1_profiles[s1_id]
                                tj = token_jaccard(s1["name_tokens"], t_toks)
                                gj = token_jaccard(s1["name_ngrams"], t_ngs)
                                if tj >= 0.40 or gj >= 0.45:
                                    cand_store[s1_id][eid] = (bname, baddr, False)
                                    s2_counts[s1_id] += 1
                                    s2_hits += 1

                # Pass 4: Location-Anchored Digit Block
                # Optimization D: Digit pre-check: only inspect digits if f1 matches and address contains a digit
                f1 = first_token(n)
                if len(f1) >= 4 and f1 in f1_in_dig and any(ch.isdigit() for ch in baddr):
                    t_digs = extract_digit_blocks(baddr)
                    if t_digs:
                        t_toks_d = None
                        for d in t_digs:
                            k_dig = (d, f1, country)
                            if k_dig in block_dig:
                                for s1_id in block_dig[k_dig]:
                                    if s2_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE and eid not in cand_store[s1_id]:
                                        if t_toks_d is None:
                                            t_toks_d = tokenize(bname)
                                        s1 = s1_profiles[s1_id]
                                        if token_overlap(s1["name_tokens"], t_toks_d) >= 1:
                                            cand_store[s1_id][eid] = (bname, baddr, False)
                                            s2_counts[s1_id] += 1
                                            s2_hits += 1

        s2_time = time.perf_counter() - t_s2_start

        # Step 3: Stream S3 (filtered for this country)
        t_s3_start = time.perf_counter()
        s3_hits = 0
        s3_scanned = 0
        with open(s3_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
            f.readline()  # Skip header
            for line in f:
                s3_scanned += 1
                prefix, tab, country_raw = line.rpartition('\t')
                if not tab or country_raw.strip() != country:
                    continue

                eid, tab2, rest = prefix.partition('\t')
                bname, tab3, baddr = rest.partition('\t')
                eid = eid.strip()

                n = normalize_business_name(bname)

                # Pass 1: Exact Name
                k_ex = (n, country) if n else None
                if k_ex and k_ex in block_exact:
                    for s1_id in block_exact[k_ex]:
                        if s3_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE:
                            if eid not in cand_store[s1_id]:
                                cand_store[s1_id][eid] = (bname, baddr, True)
                                s3_counts[s1_id] += 1
                                s3_hits += 1

                # Pass 2: Domain Stem (only if '.' is in bname)
                if '.' in bname:
                    dom_s = extract_domain_stem(bname)
                    if dom_s and len(dom_s) >= 4:
                        k_dom = (dom_s, country)
                        if k_dom in block_dom:
                            for s1_id in block_dom[k_dom]:
                                if s3_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE:
                                    if eid not in cand_store[s1_id]:
                                        cand_store[s1_id][eid] = (bname, baddr, True)
                                        s3_counts[s1_id] += 1
                                        s3_hits += 1

                # Pass 3: First-2-Tokens Prefix Match
                f2 = first_2_tokens(n)
                if len(f2) >= 4:
                    k_f2 = (f2, country)
                    if k_f2 in block_f2:
                        t_toks = None
                        t_ngs = None
                        for s1_id in block_f2[k_f2]:
                            if s3_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE and eid not in cand_store[s1_id]:
                                if t_toks is None:
                                    t_toks = tokenize(bname)
                                    t_ngs = char_ngrams(n, 3)
                                s1 = s1_profiles[s1_id]
                                tj = token_jaccard(s1["name_tokens"], t_toks)
                                gj = token_jaccard(s1["name_ngrams"], t_ngs)
                                if tj >= 0.40 or gj >= 0.45:
                                    cand_store[s1_id][eid] = (bname, baddr, True)
                                    s3_counts[s1_id] += 1
                                    s3_hits += 1

                # Pass 4: Location-Anchored Digit Block
                f1 = first_token(n)
                if len(f1) >= 4 and f1 in f1_in_dig and any(ch.isdigit() for ch in baddr):
                    t_digs = extract_digit_blocks(baddr)
                    if t_digs:
                        t_toks_d = None
                        for d in t_digs:
                            k_dig = (d, f1, country)
                            if k_dig in block_dig:
                                for s1_id in block_dig[k_dig]:
                                    if s3_counts[s1_id] < MAX_CANDIDATES_PER_SOURCE and eid not in cand_store[s1_id]:
                                        if t_toks_d is None:
                                            t_toks_d = tokenize(bname)
                                        s1 = s1_profiles[s1_id]
                                        if token_overlap(s1["name_tokens"], t_toks_d) >= 1:
                                            cand_store[s1_id][eid] = (bname, baddr, True)
                                            s3_counts[s1_id] += 1
                                            s3_hits += 1

        s3_time = time.perf_counter() - t_s3_start

        # Step 4: Feature Extraction and Model Scoring
        t_feat_start = time.perf_counter()
        country_candidate_pairs: Dict[str, List[str]] = {}
        country_matching_results: Dict[str, List[str]] = {}

        total_features_time = 0.0
        total_scoring_time = 0.0

        for s1_id in s1_order:
            cands_for_s1 = cand_store[s1_id]
            s1_prof = s1_profiles[s1_id]
            cand_pool_size = len(cands_for_s1)

            if cand_pool_size == 0:
                country_candidate_pairs[s1_id] = []
                country_matching_results[s1_id] = []
                continue

            s2_cids = sorted([cid for cid, (_, _, is_s3) in cands_for_s1.items() if not is_s3])
            s3_cids = sorted([cid for cid, (_, _, is_s3) in cands_for_s1.items() if is_s3])
            ordered_cids = s2_cids + s3_cids
            country_candidate_pairs[s1_id] = ordered_cids

            t_f0 = time.perf_counter()
            X_batch = []
            for cid in ordered_cids:
                bname, baddr, is_s3 = cands_for_s1[cid]
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
            probs = self.model.predict_proba(X_mat)[:, 1]

            max_prob = float(np.max(probs))
            if max_prob < self.tau_singleton:
                country_matching_results[s1_id] = []
            else:
                accepted = [cid for cid, p in zip(ordered_cids, probs) if p >= self.tau]
                country_matching_results[s1_id] = accepted
            total_scoring_time += (time.perf_counter() - t_m0)

        timings = {
            "index_time": idx_time,
            "s2_time": s2_time,
            "s3_time": s3_time,
            "feature_time": total_features_time,
            "score_time": total_scoring_time,
            "s2_hits": s2_hits,
            "s3_hits": s3_hits,
            "cands_count": sum(len(c) for c in country_candidate_pairs.values()),
            "matches_count": sum(len(m) for m in country_matching_results.values()),
        }

        return country_candidate_pairs, country_matching_results, timings


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
            mismatches.append(f"{label} line {idx} mismatch:\n  Optimized: {l_opt.rstrip()!r}\n  Baseline : {l_base.rstrip()!r}")
            if len(mismatches) >= 5:
                break

    return mismatches


def main():
    print("=" * 80)
    print("Starting Amazon ML Challenge 2026 — Phase 4 Step 3A: Optimized Benchmark")
    print("=" * 80)

    tracemalloc.start()
    t_global_start = time.perf_counter()

    # Step 1: Load model artifact
    engine = CountryPartitionInferenceEngine(model_artifact_path=MODEL_PATH)

    # Step 2: Load the first 1,000 S1 records
    print(f"\nLoading first {BENCHMARK_SAMPLE_SIZE:,} Source 1 records from {os.path.relpath(TEST_S1_PATH)}...")
    s1_slice = load_test_s1_slice(TEST_S1_PATH, sample_size=BENCHMARK_SAMPLE_SIZE)
    s1_order = [rec["entity_id"] for rec in s1_slice]
    print(f"Loaded {len(s1_slice):,} records.")

    # Partition by country
    s1_by_country: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for rec in s1_slice:
        s1_by_country[rec["country"]].append(rec)

    print(f"Country partition counts: { {c: len(recs) for c, recs in s1_by_country.items()} }")

    # Step 3: Run country partitions sequentially
    all_candidate_pairs: Dict[str, List[str]] = {}
    all_matching_results: Dict[str, List[str]] = {}

    tot_idx_time = 0.0
    tot_s2_time = 0.0
    tot_s3_time = 0.0
    tot_feat_time = 0.0
    tot_score_time = 0.0

    # Process countries (e.g. France, US, India)
    for country in sorted(s1_by_country.keys()):
        recs = s1_by_country[country]
        print(f"\n--- Processing Partition: {country} ({len(recs)} S1 entities) ---")
        cands, matches, timings = engine.run_country_partition(
            country=country,
            records=recs,
            s2_path=TEST_S2_PATH,
            s3_path=TEST_S3_PATH
        )

        all_candidate_pairs.update(cands)
        all_matching_results.update(matches)

        tot_idx_time += timings["index_time"]
        tot_s2_time += timings["s2_time"]
        tot_s3_time += timings["s3_time"]
        tot_feat_time += timings["feature_time"]
        tot_score_time += timings["score_time"]

        print(f"  * {country} Indexed in       : {timings['index_time']:.3f} s")
        print(f"  * {country} S2 Scanned in    : {timings['s2_time']:.2f} s ({timings['s2_hits']:,} hits)")
        print(f"  * {country} S3 Scanned in    : {timings['s3_time']:.2f} s ({timings['s3_hits']:,} hits)")
        print(f"  * {country} Features in      : {timings['feature_time']:.2f} s")
        print(f"  * {country} Model Scoring in : {timings['score_time']:.2f} s")
        print(f"  * {country} Candidates       : {timings['cands_count']:,}")
        print(f"  * {country} Accepted Matches : {timings['matches_count']:,}")

    # Step 4: Write optimized benchmark output files in EXACT original s1_order
    print("\nWriting optimized benchmark slice outputs...")
    with open(OPTIMIZED_MATCHING_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            m_list = all_matching_results.get(eid, [])
            f.write(f"{eid}\t{','.join(m_list)}\n")

    with open(OPTIMIZED_CANDIDATE_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            c_list = all_candidate_pairs.get(eid, [])
            f.write(f"{eid}\t{','.join(c_list)}\n")

    print(f"  * Output matching   : {os.path.relpath(OPTIMIZED_MATCHING_PATH)}")
    print(f"  * Output candidates : {os.path.relpath(OPTIMIZED_CANDIDATE_PATH)}")

    total_elapsed = time.perf_counter() - t_global_start
    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tot_candidates = sum(len(c) for c in all_candidate_pairs.values())
    tot_matches = sum(len(m) for m in all_matching_results.values())

    # Step 5: Exact 1:1 Output Comparison with Baseline
    print("\n" + "=" * 80)
    print("STEP H: STRICT OUTPUT VERIFICATION AGAINST BASELINE")
    print("=" * 80)

    cand_errors = compare_files_exact(
        file_opt=OPTIMIZED_CANDIDATE_PATH,
        file_base=BASELINE_CANDIDATE_PATH,
        label="Candidate Pairs"
    )
    match_errors = compare_files_exact(
        file_opt=OPTIMIZED_MATCHING_PATH,
        file_base=BASELINE_MATCHING_PATH,
        label="Matching Results"
    )

    if cand_errors or match_errors:
        print("CRITICAL VERIFICATION ERROR: Output differs from baseline!", file=sys.stderr)
        for e in cand_errors + match_errors:
            print(f"  [MISMATCH] {e}", file=sys.stderr)
        print("Stopping execution per safety instruction.", file=sys.stderr)
        sys.exit(1)

    print("SUCCESS: 100% BYTE-FOR-BYTE IDENTICAL OUTPUT TO BASELINE!")
    print("  * Same S1 IDs in identical order (1,000 entities)")
    print(f"  * Same Candidate IDs in identical order ({tot_candidates:,} candidates)")
    print(f"  * Same Final Matching Results in identical order ({tot_matches:,} matches)")
    print("=" * 80)

    # Print summary performance table
    print("\n" + "=" * 80)
    print("OPTIMIZED BENCHMARK PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Component':<32} | {'Time (s)':<12}")
    print("-" * 80)
    print(f"{'S1 Indexing Time':<32} | {tot_idx_time:<12.3f}")
    print(f"{'Source 2 Scanning Time':<32} | {tot_s2_time:<12.2f}")
    print(f"{'Source 3 Scanning Time':<32} | {tot_s3_time:<12.2f}")
    print(f"{'Feature Extraction Time':<32} | {tot_feat_time:<12.2f}")
    print(f"{'Model Scoring Time':<32} | {tot_score_time:<12.2f}")
    print(f"{'Total End-to-End Runtime':<32} | {total_elapsed:<12.2f} ({total_elapsed/60:.2f} min)")
    print(f"{'Peak Memory Footprint (RAM)':<32} | {peak_mem / (1024*1024):<12.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
