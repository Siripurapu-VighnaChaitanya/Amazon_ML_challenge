#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Phase 4 Step 3B: Full Test Inference
=============================================================================
Production script to process the full 1.73M S1 test entities.
Uses the verified parallel architecture (multiprocessing by country) to query
read-only SQLite indexes and evaluate pairs using the trained model.

Architecture:
1. Parent loads test_source1.tsv, preserves exact S1 order, and partitions by country.
2. 3 Worker processes (France, India, US) execute in parallel.
3. Workers query S2 and S3, extract features, and score with the model.
4. Workers write their results to temporary TSV files in scratch/.
5. Parent process loads the temporary TSVs and writes the final outputs
   strictly in the original S1 chronological order.
=============================================================================
"""

import collections
import concurrent.futures
import json
import multiprocessing
import os
import queue
import sqlite3
import sys
import threading
import time
import traceback
import tracemalloc
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(ANALYSIS_DIR)
sys.path.insert(0, ANALYSIS_DIR)

from phase4_features import (
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

# Output Paths
OUTPUT_CANDIDATE_PATH = os.path.join(WORKSPACE_ROOT, "output", "candidate_pairs.tsv")
OUTPUT_MATCHING_PATH = os.path.join(WORKSPACE_ROOT, "output", "matching_results.tsv")

# Reusable index storage paths
SCRATCH_DIR = os.path.join(WORKSPACE_ROOT, "scratch")
S2_INDEX_DB_PATH = os.path.join(SCRATCH_DIR, "s2_index.db")
S3_INDEX_DB_PATH = os.path.join(SCRATCH_DIR, "s3_index.db")

MAX_CANDIDATES_PER_SOURCE = 30
REPORT_BATCH_SIZE = 5000


def load_test_s1_full(file_path: str, limit: int = None) -> List[Dict[str, str]]:
    """Load Source 1 test records."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()  # Skip header
        for line in f:
            if limit and len(records) >= limit:
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

    # Capping by chronological order
    sorted_rids = sorted(cand_store.keys())[:max_cands]
    return [cand_store[rid] for rid in sorted_rids]


def worker_task(country: str, s1_slice: List[Dict[str, str]], q: multiprocessing.Queue) -> Dict[str, Any]:
    """
    Worker task: Processes S1 entities for a specific country and writes results to temporary TSVs.
    """
    t_start = time.perf_counter()
    
    try:
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

        # Setup temp output files
        temp_cand_path = os.path.join(SCRATCH_DIR, f"temp_cands_{country}.tsv")
        temp_match_path = os.path.join(SCRATCH_DIR, f"temp_match_{country}.tsv")
        
        f_cand = open(temp_cand_path, 'w', encoding='utf-8')
        f_match = open(temp_match_path, 'w', encoding='utf-8')

        processed = 0
        cands_generated = 0
        matches_found = 0
        
        for rec in s1_slice:
            s1_id = rec["entity_id"]
            s1_prof = build_record_profile(rec["name"], rec["addr"], rec["country"])
            
            # Query S2 and S3
            s2_matches = query_source_index(conn_s2, s1_prof, max_cands=MAX_CANDIDATES_PER_SOURCE)
            s3_matches = query_source_index(conn_s3, s1_prof, max_cands=MAX_CANDIDATES_PER_SOURCE)
            
            cand_dict = {}
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
            
            # Write Candidates
            cids_str = ",".join(cids_ordered)
            f_cand.write(f"{s1_id}\t{cids_str}\n")
            
            # Scoring
            if not cids_ordered:
                f_match.write(f"{s1_id}\t\n")
            else:
                X = []
                for cid in cids_ordered:
                    bname, baddr, is_s3 = cand_dict[cid]
                    cand_prof = build_record_profile(bname, baddr, country)
                    X.append(extract_feature_vector(s1_prof, cand_prof, is_s3, len(cids_ordered)))
                    
                probs = model.predict_proba(np.array(X, dtype=np.float32))[:, 1]
                max_p = float(np.max(probs))
                
                if max_p < tau_singleton:
                    f_match.write(f"{s1_id}\t\n")
                else:
                    accepted = [c for c, p in zip(cids_ordered, probs) if p >= tau]
                    matches_found += len(accepted)
                    acc_str = ",".join(accepted)
                    f_match.write(f"{s1_id}\t{acc_str}\n")

            processed += 1
            cands_generated += len(cids_ordered)

            if processed % REPORT_BATCH_SIZE == 0:
                q.put({
                    "type": "progress",
                    "country": country,
                    "processed": processed,
                    "cands": cands_generated,
                    "matches": matches_found,
                    "elapsed": time.perf_counter() - t_start
                })
                
        conn_s2.close()
        conn_s3.close()
        f_cand.close()
        f_match.close()
        
        t_elapsed = time.perf_counter() - t_start
        q.put({
            "type": "complete",
            "country": country,
            "processed": processed,
            "cands": cands_generated,
            "matches": matches_found,
            "elapsed": t_elapsed
        })
        
        return {
            "country": country,
            "elapsed": t_elapsed,
            "cand_path": temp_cand_path,
            "match_path": temp_match_path,
            "processed": processed
        }
    except Exception as e:
        q.put({
            "type": "error",
            "country": country,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        raise


def progress_listener(q: multiprocessing.Queue, total_counts: Dict[str, int]):
    log_path = os.path.join(SCRATCH_DIR, "step3b_progress.log")
    json_path = os.path.join(SCRATCH_DIR, "step3b_progress.json")
    
    status_data = {
        "status": "RUNNING",
        "overall": {"processed": 0, "total": sum(total_counts.values()), "percentage": 0.0},
        "countries": {c: {"processed": 0, "total": t, "percentage": 0.0, "status": "RUNNING"} for c, t in total_counts.items()},
        "total_candidates": 0,
        "total_matches": 0,
        "elapsed_seconds": 0.0
    }
    
    def write_json():
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
            
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] Started Inference Pipeline\n")
    write_json()
    
    t_start = time.perf_counter()
    last_print = t_start
    
    while True:
        try:
            msg = q.get(timeout=2.0)
            if msg is None:  # Shutdown signal
                if status_data["status"] == "RUNNING":
                    status_data["status"] = "COMPLETE"
                write_json()
                break
                
            if msg["type"] in ("progress", "complete"):
                country = msg["country"]
                proc = msg["processed"]
                total = total_counts[country]
                pct = (proc / total) * 100 if total else 0
                
                status_data["countries"][country]["processed"] = proc
                status_data["countries"][country]["percentage"] = pct
                status_data["countries"][country]["cands"] = msg["cands"]
                status_data["countries"][country]["matches"] = msg["matches"]
                
                if msg["type"] == "complete":
                    status_data["countries"][country]["status"] = "COMPLETE"
                
                # Recompute overall
                ov_proc = sum(v["processed"] for v in status_data["countries"].values())
                ov_tot = status_data["overall"]["total"]
                status_data["overall"]["processed"] = ov_proc
                status_data["overall"]["percentage"] = (ov_proc / ov_tot) * 100 if ov_tot else 0
                status_data["elapsed_seconds"] = time.perf_counter() - t_start
                status_data["total_candidates"] = sum(v.get("cands", 0) for v in status_data["countries"].values())
                status_data["total_matches"] = sum(v.get("matches", 0) for v in status_data["countries"].values())
                
                write_json()
                
                if msg["type"] == "progress":
                    elap_str = time.strftime("%H:%M:%S", time.gmtime(msg["elapsed"]))
                    timestamp = time.strftime("%H:%M:%S")
                    log_line = f"[{timestamp}] {country} | {proc} / {total} | {pct:.2f}% | candidates={status_data['total_candidates']} | matches={status_data['total_matches']} | elapsed={elap_str}\n"
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(log_line)
                        
            elif msg["type"] == "error":
                status_data["status"] = "ERROR"
                status_data["error_details"] = msg
                write_json()
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n[{time.strftime('%H:%M:%S')}] ERROR in {msg['country']}: {msg['error']}\n{msg['traceback']}\n")
                
        except queue.Empty:
            pass
            
        # Periodic console reporting (every 30 seconds)
        now = time.perf_counter()
        if now - last_print > 30.0:
            ov = status_data["overall"]
            print(f"\n--- Progress Update ---")
            print(f"Overall: {ov['processed']:,} / {ov['total']:,} ({ov['percentage']:.2f}%)")
            for c, cd in status_data["countries"].items():
                print(f"  {c}: {cd['processed']:,} / {cd['total']:,}")
            print(f"Total Candidates: {status_data['total_candidates']:,}")
            print(f"Total Matches: {status_data['total_matches']:,}")
            print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(now - t_start))}")
            last_print = now


def main():
    print("=" * 80)
    print("Starting Amazon ML Challenge 2026 — Phase 4 Step 3B: FULL Parallel Inference")
    print("=" * 80)

    # For testing, we can limit the rows
    LIMIT = None  
    if "--test" in sys.argv:
        LIMIT = 20000
        print("Running in TEST mode (Limited rows)")

    tracemalloc.start()
    t_global_start = time.perf_counter()

    t_s1_start = time.perf_counter()
    print(f"\nLoading Source 1 records from {os.path.relpath(TEST_S1_PATH)}...")
    s1_slice = load_test_s1_full(TEST_S1_PATH, limit=LIMIT)
    s1_order = [rec["entity_id"] for rec in s1_slice]
    print(f"Loaded {len(s1_slice):,} records.")
    
    country_partitions = collections.defaultdict(list)
    for rec in s1_slice:
        country_partitions[rec["country"]].append(rec)
        
    total_counts = {c: len(items) for c, items in country_partitions.items()}
    print("Partitions:")
    for country, count in total_counts.items():
        print(f"  {country}: {count:,} entities")

    m = multiprocessing.Manager()
    q = m.Queue()
    
    monitor_thread = threading.Thread(target=progress_listener, args=(q, total_counts))
    monitor_thread.start()

    print(f"\nProcessing {len(s1_order):,} Source 1 entities across {len(country_partitions)} workers...")
    t_query_start = time.perf_counter()
    worker_results = []
    has_error = False

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(country_partitions)) as executor:
        futures = {
            executor.submit(worker_task, country, items, q): country
            for country, items in country_partitions.items()
        }
        
        for future in concurrent.futures.as_completed(futures):
            country = futures[future]
            try:
                res = future.result()
                worker_results.append(res)
            except Exception as e:
                print(f"CRITICAL ERROR: Worker [{country}] failed! Terminating.")
                has_error = True
                executor.shutdown(wait=False, cancel_futures=True)
                break

    q.put(None)  # shutdown signal
    monitor_thread.join()

    if has_error:
        print("Pipeline aborted due to worker error. Check scratch/step3b_progress.log for details.")
        sys.exit(1)

    processing_time = time.perf_counter() - t_query_start
    print(f"\nParallel processing completed in {processing_time:.2f} s.")

    print("\nMerging temporary worker files in exact S1 order...")
    t_merge_start = time.perf_counter()
    
    cand_dict = {}
    match_dict = {}
    
    for res in worker_results:
        with open(res["cand_path"], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                cand_dict[parts[0]] = parts[1] if len(parts) > 1 else ""
        with open(res["match_path"], 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                match_dict[parts[0]] = parts[1] if len(parts) > 1 else ""

    missing = set(s1_order) - set(cand_dict.keys())
    if missing:
        print(f"CRITICAL ERROR: {len(missing)} S1 entities missing from worker outputs!", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_MATCHING_PATH), exist_ok=True)
    with open(OUTPUT_MATCHING_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_order:
            f.write(f"{eid}\t{match_dict[eid]}\n")

    with open(OUTPUT_CANDIDATE_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            f.write(f"{eid}\t{cand_dict[eid]}\n")
            
    merge_time = time.perf_counter() - t_merge_start
    print(f"Merge completed in {merge_time:.2f} s.")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.perf_counter() - t_global_start

    try:
        out_cand_size = os.path.getsize(OUTPUT_CANDIDATE_PATH)
        out_match_size = os.path.getsize(OUTPUT_MATCHING_PATH)
    except Exception:
        out_cand_size = 0
        out_match_size = 0

    print("\n" + "=" * 80)
    print("PHASE 4 STEP 3B PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Total Processed S1 Entities: {len(s1_order):,}")
    print(f"Total Runtime: {total_time:.2f} s ({total_time/3600:.2f} hours)")
    print(f"Peak Process RAM (Parent Only): {peak_mem / (1024*1024):.2f} MB")
    print(f"Output Candidates: {OUTPUT_CANDIDATE_PATH} ({out_cand_size / (1024*1024):.2f} MB)")
    print(f"Output Matches: {OUTPUT_MATCHING_PATH} ({out_match_size / (1024*1024):.2f} MB)")
    print("=" * 80)
    
    # Append final status to JSON and Log
    json_path = os.path.join(SCRATCH_DIR, "step3b_progress.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
        status_data["status"] = "COMPLETE"
        status_data["total_runtime_seconds"] = total_time
        status_data["output_cand_bytes"] = out_cand_size
        status_data["output_match_bytes"] = out_match_size
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)
            
    log_path = os.path.join(SCRATCH_DIR, "step3b_progress.log")
    if os.path.exists(log_path):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[{time.strftime('%H:%M:%S')}] STATUS = COMPLETE\n")
            f.write(f"Total Processed: {len(s1_order):,}\n")
            f.write(f"Total Candidates: {status_data.get('total_candidates', 0):,}\n")
            f.write(f"Total Matches: {status_data.get('total_matches', 0):,}\n")
            f.write(f"Total Runtime: {total_time:.2f} s\n")
            f.write(f"Output Candidates Size: {out_cand_size / (1024*1024):.2f} MB\n")
            f.write(f"Output Matches Size: {out_match_size / (1024*1024):.2f} MB\n")


if __name__ == "__main__":
    main()
