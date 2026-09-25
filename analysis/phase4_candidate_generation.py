#!/usr/bin/env python3
"""
Phase 4 Step 1: Candidate Generation Module
Amazon ML Challenge 2026 — Business Entity Resolution

This module implements the decoupled Stage 1 Candidate Generator.
It uses the multi-pass blocking discoveries from Phase 3B without applying
the downstream heuristic matching rules, preserving candidates for the ML reranker:

1. Hard Country Partition (zero cross-country candidate pairs)
2. Exact Normalized Name Index
3. Web Domain Stem Index (.com, .in, .fr, .org, etc.)
4. First-2-Tokens Prefix Index + High Fuzzy Name Similarity
5. Location-Anchored Digit Block + First Token Index

Key Principles:
- Complete decoupling of candidate generation from final classification decisions.
- Candidate deduplication per Source 1 entity.
- Source 2 and Source 3 identity preservation (S2-, S3- prefixes).
- Zero access to train_ground_truth.tsv during candidate generation.
- Deterministic training sample selection (30,000 IDs, seed 43, disjoint from validation).
- Streaming/chunked execution targeting < 500 MB memory footprint.
"""

import os
import sys
import time
import random
import collections
import tracemalloc
from typing import Dict, Set, List, Tuple, Any, Optional, Iterator

# Ensure analysis directory is on python path for importing phase4_features
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
    extract_feature_dict,
    FEATURE_NAMES,
)

# Standard file paths
DATASET_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
DATASET_TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

TRAIN_S1_PATH = os.path.join(DATASET_TRAIN_DIR, "train_source1.tsv")
TRAIN_S2_PATH = os.path.join(DATASET_TRAIN_DIR, "train_source2.tsv")
TRAIN_S3_PATH = os.path.join(DATASET_TRAIN_DIR, "train_source3.tsv")
TRAIN_GT_PATH = os.path.join(DATASET_TRAIN_DIR, "train_ground_truth.tsv")

VAL_IDS_PATH = os.path.join(CURRENT_DIR, "phase2_validation_ids.txt")

DEFAULT_TRAIN_SAMPLE_SIZE = 30000
DEFAULT_TRAIN_SEED = 43


def first_2_tokens(text: str) -> str:
    """Extract first 2 tokens of string."""
    toks = text.split()
    return ' '.join(toks[:2]) if len(toks) >= 2 else (toks[0] if toks else '')


def first_token(text: str) -> str:
    """Extract first token of string."""
    toks = text.split()
    return toks[0] if toks else ''


def select_training_s1_ids(
    s1_path: str = TRAIN_S1_PATH,
    val_ids_path: str = VAL_IDS_PATH,
    sample_size: int = DEFAULT_TRAIN_SAMPLE_SIZE,
    seed: int = DEFAULT_TRAIN_SEED
) -> List[str]:
    """
    Selects exactly sample_size Source 1 IDs from s1_path,
    deterministically using seed, strictly disjoint from val_ids_path.
    """
    if not os.path.isfile(val_ids_path):
        raise FileNotFoundError(f"Validation IDs file not found: {val_ids_path}")

    with open(val_ids_path, 'r', encoding='utf-8') as f:
        val_set = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(val_set):,} validation IDs from {os.path.relpath(val_ids_path)}")

    eligible_ids = []
    with open(s1_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
        f.readline()  # Skip header
        for line in f:
            eid = line.split('\t', 1)[0].strip()
            if eid and eid not in val_set:
                eligible_ids.append(eid)

    print(f"Total eligible non-validation Source 1 IDs: {len(eligible_ids):,}")
    if len(eligible_ids) < sample_size:
        raise ValueError(
            f"Requested {sample_size:,} IDs, but only {len(eligible_ids):,} eligible IDs exist."
        )

    rng = random.Random(seed)
    sampled = sorted(rng.sample(eligible_ids, sample_size))
    print(f"Deterministically sampled {len(sampled):,} training S1 IDs (seed={seed})")

    # Safety verification
    overlap = set(sampled) & val_set
    assert len(overlap) == 0, f"FATAL: Overlap detected between train and val IDs: {len(overlap)}"
    return sampled


class CandidateGenerator:
    """
    Reusable Stage 1 Candidate Generator.

    Builds country-partitioned multi-pass blocking indices over Source 1 entities
    and streams Source 2 and Source 3 files to produce deduplicated candidate pairs.
    """

    def __init__(self, max_candidates_per_entity: Optional[int] = None):
        """
        Args:
            max_candidates_per_entity: Optional safety ceiling on candidates per entity.
                                      None means unconstrained.
        """
        self.max_candidates = max_candidates_per_entity
        self.s1_profiles: Dict[str, Dict[str, Any]] = {}
        self.block_exact: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        self.block_f2: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        self.block_dom: Dict[Tuple[str, str], List[str]] = collections.defaultdict(list)
        self.block_dig: Dict[Tuple[str, str, str], List[str]] = collections.defaultdict(list)

    def build_index(self, s1_records: Dict[str, Dict[str, str]]) -> None:
        """
        Build multi-pass blocking indices from a dictionary of Source 1 records.

        Args:
            s1_records: dict of s1_id -> {'name': str, 'addr': str, 'country': str}
        """
        self.s1_profiles.clear()
        self.block_exact.clear()
        self.block_f2.clear()
        self.block_dom.clear()
        self.block_dig.clear()

        for eid, rec in s1_records.items():
            name = rec.get("name", "")
            addr = rec.get("addr", "")
            country = rec.get("country", "").strip()

            profile = build_record_profile(name, addr, country)
            self.s1_profiles[eid] = profile

            n = profile["norm_name"]
            stripped = profile["stripped_name"]
            digs = profile["digits"]

            # Pass 1: Exact Normalized Name Block (Country scoped)
            if n:
                self.block_exact[(n, country)].append(eid)

            # Pass 2: First-2-Tokens Prefix Block (Country scoped)
            f2 = first_2_tokens(n)
            if len(f2) >= 4:
                self.block_f2[(f2, country)].append(eid)

            # Pass 3: Domain Stem Block (Country scoped)
            if len(stripped) >= 4:
                self.block_dom[(stripped, country)].append(eid)

            # Pass 4: Street/Postal Digit + First Token Block (Country scoped)
            f1 = profile["first_tok"]
            if len(f1) >= 4 and digs:
                for d in digs:
                    self.block_dig[(d, f1, country)].append(eid)

    def stream_candidates(
        self,
        s2_path: str,
        s3_path: str,
        max_s2_rows: Optional[int] = None,
        max_s3_rows: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Stream Source 2 and Source 3 files against the built indices.

        Returns:
            Dict mapping s1_id -> list of deduplicated candidate IDs (ordered S2 then S3).
        """
        # Initialize candidate sets per S1 entity
        candidates: Dict[str, Dict[str, Set[str]]] = {
            s1_id: {"s2": set(), "s3": set()}
            for s1_id in self.s1_profiles
        }

        sources = [("S2", s2_path, max_s2_rows), ("S3", s3_path, max_s3_rows)]

        for src_label, src_path, max_rows in sources:
            if not os.path.isfile(src_path):
                print(f"Warning: {src_path} not found. Skipping {src_label}.", file=sys.stderr)
                continue

            src_key = src_label.lower()
            scanned = 0

            with open(src_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
                f.readline()  # Skip header
                for line in f:
                    if max_rows and scanned >= max_rows:
                        break

                    parts = line.rstrip('\r\n').split('\t')
                    if len(parts) >= 4:
                        scanned += 1
                        eid = parts[0].strip()
                        bname = parts[1]
                        baddr = parts[2]
                        country = parts[3].strip()

                        n = normalize_business_name(bname)

                        # Lookup keys
                        k_ex = (n, country) if n else None
                        f2 = first_2_tokens(n)
                        k_f2 = (f2, country) if len(f2) >= 4 else None
                        dom_s = extract_domain_stem(bname)
                        k_dom = (dom_s, country) if (dom_s and len(dom_s) >= 4) else None

                        # Lazy feature evaluation for fuzzy/relational passes
                        t_toks = None
                        t_ngs = None
                        t_digs = None

                        # Pass 1: Exact Normalized Name Match
                        if k_ex and k_ex in self.block_exact:
                            for s1_id in self.block_exact[k_ex]:
                                if self.max_candidates is None or len(candidates[s1_id][src_key]) < self.max_candidates:
                                    candidates[s1_id][src_key].add(eid)

                        # Pass 2: Domain Stem Match
                        if k_dom and k_dom in self.block_dom:
                            for s1_id in self.block_dom[k_dom]:
                                if self.max_candidates is None or len(candidates[s1_id][src_key]) < self.max_candidates:
                                    candidates[s1_id][src_key].add(eid)

                        # Pass 3: First-2-Tokens Prefix Match (Fuzzy Name Candidate)
                        if k_f2 and k_f2 in self.block_f2:
                            if t_toks is None:
                                t_toks = tokenize(bname)
                                t_ngs = char_ngrams(n, 3)

                            for s1_id in self.block_f2[k_f2]:
                                if eid not in candidates[s1_id][src_key]:
                                    s1 = self.s1_profiles[s1_id]
                                    tj = token_jaccard(s1["name_tokens"], t_toks)
                                    gj = token_jaccard(s1["name_ngrams"], t_ngs)

                                    # Permissive candidate threshold (tj >= 0.40 or gj >= 0.45)
                                    # Captures typos and spelling variants without address filtering
                                    if tj >= 0.40 or gj >= 0.45:
                                        if self.max_candidates is None or len(candidates[s1_id][src_key]) < self.max_candidates:
                                            candidates[s1_id][src_key].add(eid)

                        # Pass 4: Location-Anchored Digit Block + First Token Match
                        f1 = first_token(n)
                        if len(f1) >= 4:
                            if t_digs is None:
                                t_digs = extract_digit_blocks(baddr)
                            if t_digs:
                                for d in t_digs:
                                    k_dig = (d, f1, country)
                                    if k_dig in self.block_dig:
                                        if t_toks is None:
                                            t_toks = tokenize(bname)

                                        for s1_id in self.block_dig[k_dig]:
                                            if eid not in candidates[s1_id][src_key]:
                                                s1 = self.s1_profiles[s1_id]
                                                # Minimum token overlap to filter complete noise
                                                if token_overlap(s1["name_tokens"], t_toks) >= 1:
                                                    if self.max_candidates is None or len(candidates[s1_id][src_key]) < self.max_candidates:
                                                        candidates[s1_id][src_key].add(eid)

        # Merge deduplicated candidates per S1 entity (S2 sorted then S3 sorted)
        result: Dict[str, List[str]] = {}
        for s1_id, pool in candidates.items():
            s2_sorted = sorted(pool["s2"])
            s3_sorted = sorted(pool["s3"])
            result[s1_id] = s2_sorted + s3_sorted

        return result


def load_s1_records_by_ids(
    s1_path: str,
    target_ids: Set[str]
) -> Dict[str, Dict[str, str]]:
    """
    Stream Source 1 file and extract only records whose entity_id is in target_ids.
    """
    records = {}
    with open(s1_path, 'r', encoding='utf-8', buffering=1024 * 1024) as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 4 and parts[0] in target_ids:
                records[parts[0]] = {
                    "name": parts[1],
                    "addr": parts[2],
                    "country": parts[3].strip()
                }
            if len(records) == len(target_ids):
                break
    return records


# ==============================================================================
# Self-Test and Sanity Verification Suite
# ==============================================================================
def run_unit_tests():
    """Verify feature extraction and candidate generation functionality."""
    print("=" * 80)
    print("Running Phase 4 Step 1 Verification Suite")
    print("=" * 80)

    # Test 1: Feature Vector Dimensionality & Ordering
    print("\n--- Test 1: Feature Vector Specification ---")
    assert len(FEATURE_NAMES) == 18, f"Expected 18 feature names, got {len(FEATURE_NAMES)}"
    print(f"Feature count verified: {len(FEATURE_NAMES)} features.")

    # Test 2: Legal Suffix Handling (US, India, France)
    print("\n--- Test 2: Legal Suffix Normalization ---")
    test_cases = [
        ("Acme Corporation", "acme"),
        ("Tata Consultancy Services Pvt Ltd", "tata consultancy services"),
        ("Marina Ecole France SARL", "marina ecole france"),
        ("SCI Ptit Amicale", "sci ptit amicale"),  # SCI at start preserved
        ("ZNB Club SAS", "znb club"),
        ("Alpha Beta LLC", "alpha beta"),
        ("Gamma Delta Limited", "gamma delta"),
    ]
    for raw, expected in test_cases:
        norm = normalize_business_name(raw)
        assert norm == expected, f"Expected '{expected}', got '{norm}' for '{raw}'"
    print("Legal suffix stripping passed across US, India, and France!")

    # Test 3: Domain Stem & Digit Extraction
    print("\n--- Test 3: Domain Stem and Digit Extraction ---")
    assert extract_domain_stem("google.com") == "google"
    assert extract_domain_stem("infosys.in") == "infosys"
    assert extract_domain_stem("leboncoin.fr") == "leboncoin"
    assert extract_domain_stem("No domain name") == ""

    addr = "Plot 448A, Udyog Vihar Phase V, Gurugram 122016, HR"
    digs = extract_digit_blocks(addr)
    assert "448" in digs and "122016" in digs
    print("Domain stem and digit block extraction passed!")

    # Test 4: Feature Extraction and Value Ranges
    print("\n--- Test 4: Feature Extraction and Range Invariants ---")
    r1 = {
        "raw_name": "Zephay Labs Inc",
        "raw_addr": "2621 Cotten Road, Tyler, TX 75701",
        "country": "US"
    }
    r2 = {
        "raw_name": "Zephay Labs LLC",
        "raw_addr": "2621 Cotten Rd, Tyler, Texas 75701",
        "country": "US"
    }
    f_vec = extract_feature_vector(r1, r2, is_source3=False, cand_pool_size=5)
    f_dict = extract_feature_dict(r1, r2, is_source3=False, cand_pool_size=5)

    assert len(f_vec) == 18, f"Feature vector length must be 18, got {len(f_vec)}"
    assert f_dict["name_exact"] == 1.0  # zephay labs == zephay labs
    assert f_dict["name_tok_jaccard"] == 1.0
    assert f_dict["addr_tok_jaccard"] > 0.3
    assert f_dict["shared_digit_count"] >= 2.0  # 2621 and 75701
    assert f_dict["has_shared_digits"] == 1.0
    assert f_dict["conflicting_digits"] == 0.0
    assert f_dict["is_source3"] == 0.0
    assert f_dict["cand_pool_size"] == 5.0

    # Range verification
    for name, val in f_dict.items():
        assert isinstance(val, (float, int)), f"Feature {name} is not numeric: {type(val)}"
        if "jaccard" in name or "ratio" in name or "containment" in name:
            assert 0.0 <= val <= 1.0, f"Feature {name} out of [0, 1] range: {val}"
        if name in ("name_exact", "name_first_tok_match", "addr_missing_either",
                    "has_shared_digits", "conflicting_digits", "domain_stem_match", "is_source3"):
            assert val in (0.0, 1.0), f"Binary feature {name} must be 0 or 1: {val}"
    print("Feature vector ranges and invariants verified successfully!")

    # Test 5: Training Sample Selection (Deterministic 30,000 IDs, Disjoint from Val)
    print("\n--- Test 5: Training Sample Selection (30,000 IDs, Seed 43) ---")
    t0 = time.perf_counter()
    train_ids = select_training_s1_ids(
        s1_path=TRAIN_S1_PATH,
        val_ids_path=VAL_IDS_PATH,
        sample_size=DEFAULT_TRAIN_SAMPLE_SIZE,
        seed=DEFAULT_TRAIN_SEED
    )
    t_sample = time.perf_counter() - t0

    assert len(train_ids) == DEFAULT_TRAIN_SAMPLE_SIZE, f"Expected {DEFAULT_TRAIN_SAMPLE_SIZE} IDs"

    with open(VAL_IDS_PATH, 'r', encoding='utf-8') as f:
        val_ids = {line.strip() for line in f if line.strip()}

    overlap = set(train_ids) & val_ids
    assert len(overlap) == 0, f"Disjointness violated! Found {len(overlap)} overlapping IDs"
    print(f"Selected {len(train_ids):,} training S1 IDs in {t_sample:.2f}s.")
    print(f"Validation overlap: {len(overlap)} IDs (Strictly 100% disjoint).")

    # Test 6: Controlled Candidate Generation on Small Sample (50 S1 entities against 5,000 S2/S3 rows)
    print("\n--- Test 6: Controlled Candidate Generation on Small Slice ---")
    small_s1_ids = set(train_ids[:50])
    small_s1_records = load_s1_records_by_ids(TRAIN_S1_PATH, small_s1_ids)
    assert len(small_s1_records) == 50

    generator = CandidateGenerator()
    generator.build_index(small_s1_records)

    t0 = time.perf_counter()
    cand_results = generator.stream_candidates(
        s2_path=TRAIN_S2_PATH,
        s3_path=TRAIN_S3_PATH,
        max_s2_rows=5000,
        max_s3_rows=5000
    )
    t_cand = time.perf_counter() - t0

    total_cands = sum(len(c_list) for c_list in cand_results.values())
    print(f"Controlled candidate generation completed in {t_cand:.2f}s.")
    print(f"50 S1 entities evaluated: {total_cands} total candidate pairs generated.")

    # Verify ID formatting and deduplication
    for s1_id, c_list in cand_results.items():
        assert len(c_list) == len(set(c_list)), f"Candidate list for {s1_id} contains duplicates!"
        for cid in c_list:
            assert cid.startswith(("S2-", "S3-")), f"Invalid candidate ID prefix: {cid}"

    print("Candidate generation deduplication and ID prefix checks passed!")
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_tests()
