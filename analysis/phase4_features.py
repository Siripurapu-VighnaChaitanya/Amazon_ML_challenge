#!/usr/bin/env python3
"""
Phase 4 Step 1: Feature Engineering Module
Amazon ML Challenge 2026 — Business Entity Resolution

This module implements the exact 18-dimensional feature vector specified in the
Phase 4 architecture plan. All features are strictly pairwise or candidate-metadata
derived, with zero data leakage and zero use of ground-truth labels.

Features:
 1. name_exact              (Binary)   Exact normalized business name equality
 2. name_tok_jaccard        (Float)    Word token Jaccard similarity of names
 3. name_tok_overlap        (Float)    Shared token count between names
 4. name_tok_containment    (Float)    Token overlap divided by min token count
 5. name_ngram_jaccard      (Float)    Character 3-gram Jaccard similarity of names
 6. name_levenshtein_ratio  (Float)    Normalized Levenshtein edit distance ratio
 7. name_first_tok_match    (Binary)   First word token match between names
 8. name_len_diff_ratio     (Float)    Normalized name length difference
 9. addr_tok_jaccard        (Float)    Word token Jaccard similarity of addresses
10. addr_tok_overlap        (Float)    Shared token count between addresses
11. addr_tok_containment    (Float)    Address token overlap divided by min token count
12. addr_missing_either     (Binary)   1 if either address is empty/whitespace
13. shared_digit_count      (Float)    Count of shared 3-6 digit sequences (house/PIN)
14. has_shared_digits       (Binary)   1 if shared_digit_count > 0
15. conflicting_digits      (Binary)   1 if both have digits but zero intersection
16. domain_stem_match       (Binary)   1 if extracted web domain stem matches name
17. is_source3              (Binary)   1 if candidate is from Source 3, 0 for Source 2
18. cand_pool_size          (Float)    Total candidate count for this Source 1 entity

Supports US, India, and France entities. Pure standard library + NumPy compatible.
"""

import re
import unicodedata
from typing import Dict, Set, List, Tuple, Any, Optional

FEATURE_NAMES: List[str] = [
    "name_exact",
    "name_tok_jaccard",
    "name_tok_overlap",
    "name_tok_containment",
    "name_ngram_jaccard",
    "name_levenshtein_ratio",
    "name_first_tok_match",
    "name_len_diff_ratio",
    "addr_tok_jaccard",
    "addr_tok_overlap",
    "addr_tok_containment",
    "addr_missing_either",
    "shared_digit_count",
    "has_shared_digits",
    "conflicting_digits",
    "domain_stem_match",
    "is_source3",
    "cand_pool_size",
]

# Legal suffixes across US, India, and France
# US/Common: corp, corporation, inc, incorporated, llc, llp, pllc, co, company, ltd, limited
# India: private limited, pvt ltd, pvt limited, private ltd
# France: sarl, sas, sa, sci, eurl, snc, gie, sca, association
LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b('
    r'private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|'
    r'corporation|corp|incorporated|inc|limited|ltd|llc|llp|pllc|'
    r'sarl|sas|sci|eurl|snc|gie|sca|association|'
    r'company|co'
    r')\b\s*$',
    re.IGNORECASE
)

# Web domain stem extraction supporting .com, .in, .org, .net, .fr, .co, .io, .biz, .info, .edu, .gov
DOMAIN_PATTERN = re.compile(
    r'\b([a-z0-9\-]+)\.(com|in|org|net|fr|co|io|biz|info|edu|gov)\b',
    re.IGNORECASE
)

# Street/house numbers, PIN codes, postal codes (3 to 6 digits)
DIGIT_BLOCK_PATTERN = re.compile(r'(?<!\d)\d{3,6}(?!\d)')


def normalize_text(text: str) -> str:
    """Normalize text with NFKD decomposition and ASCII transliteration."""
    if not text:
        return ""
    decomposed = unicodedata.normalize('NFKD', text)
    lowered = decomposed.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', lowered)
    return ' '.join(cleaned.split())


def normalize_business_name(name: str) -> str:
    """
    Standardized business name normalization:
    1. NFKD decomposition and lowercase conversion
    2. Punctuation removal (replaced with space)
    3. Iterative legal suffix stripping (US, India, France)
    4. Whitespace collapsing and trimming
    """
    if not name:
        return ""
    text = unicodedata.normalize('NFKD', name).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    curr = text.strip()
    prev = None
    while curr != prev:
        prev = curr
        stripped = LEGAL_SUFFIX_PATTERN.sub('', curr).strip()
        if len(stripped) >= 3:
            curr = stripped
        else:
            break
    return ' '.join(curr.split())


def tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return set()
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return {w for w in cleaned.split() if w}


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from whitespace-padded string."""
    if not text:
        return set()
    cleaned = re.sub(r'\s+', ' ', text.lower().strip())
    padded = f"  {cleaned}  "
    if len(padded) < n:
        return set()
    return {padded[i:i + n] for i in range(len(padded) - n + 1)}


def token_jaccard(s1: Set[Any], s2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return float(intersection / union) if union > 0 else 0.0


def token_overlap(s1: Set[Any], s2: Set[Any]) -> float:
    """Compute absolute overlap count between two sets."""
    if not s1 or not s2:
        return 0.0
    return float(len(s1 & s2))


def token_containment(s1: Set[Any], s2: Set[Any]) -> float:
    """Compute token containment: overlap divided by min(len(s1), len(s2))."""
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    return float(len(s1 & s2) / min_len)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    Memory-efficient two-row DP implementation in O(min(M, N)) space.
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    # len(s1) >= len(s2)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Normalized Levenshtein similarity ratio in [0.0, 1.0].
    Returns 1.0 for identical strings, 0.0 for completely disjoint.
    """
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return float(max(0.0, 1.0 - (dist / max_len)))


def extract_digit_blocks(text: str) -> Set[str]:
    """Extract street numbers, PIN codes, and postal codes (3-6 digits)."""
    if not text:
        return set()
    return set(DIGIT_BLOCK_PATTERN.findall(text))


def extract_domain_stem(text: str) -> str:
    """If text contains a domain name (e.g. xyz.com, abc.fr), extract stem (xyz, abc)."""
    if not text:
        return ""
    match = DOMAIN_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return ""


def first_token(text: str) -> str:
    """Extract first token of string."""
    toks = text.split()
    return toks[0] if toks else ""


def build_record_profile(
    name: str,
    addr: str,
    country: str = ""
) -> Dict[str, Any]:
    """
    Precompute reusable text features for a record to avoid repeated computation.
    """
    raw_name = name or ""
    raw_addr = addr or ""
    norm_name = normalize_business_name(raw_name)
    name_toks = tokenize(norm_name) if norm_name else tokenize(raw_name)
    name_ngs = char_ngrams(norm_name, 3)
    addr_toks = tokenize(raw_addr)
    digits = extract_digit_blocks(raw_addr)
    dom_stem = extract_domain_stem(raw_name)
    stripped_name = re.sub(r'[^a-z0-9]', '', norm_name)
    f_tok = first_token(norm_name)

    return {
        "raw_name": raw_name,
        "raw_addr": raw_addr,
        "norm_name": norm_name,
        "name_tokens": name_toks,
        "name_ngrams": name_ngs,
        "addr_tokens": addr_toks,
        "digits": digits,
        "domain_stem": dom_stem,
        "stripped_name": stripped_name,
        "first_tok": f_tok,
        "country": (country or "").strip(),
        "is_addr_empty": (not raw_addr.strip()),
    }


def extract_feature_vector(
    rec1: Dict[str, Any],
    rec2: Dict[str, Any],
    is_source3: bool = False,
    cand_pool_size: int = 1
) -> List[float]:
    """
    Extract the exact 18-dimensional numeric feature vector for a candidate pair.

    Args:
        rec1: Profile dict for Source 1 record (from build_record_profile or dict).
        rec2: Profile dict for Candidate record (Source 2 or Source 3).
        is_source3: True if candidate is from Source 3, False if from Source 2.
        cand_pool_size: Total candidate count generated for this Source 1 entity.

    Returns:
        18-dimensional list of floats matching FEATURE_NAMES order.
    """
    n1 = rec1.get("norm_name") or normalize_business_name(rec1.get("raw_name", ""))
    n2 = rec2.get("norm_name") or normalize_business_name(rec2.get("raw_name", ""))

    toks1 = rec1.get("name_tokens") if "name_tokens" in rec1 else (tokenize(n1) if n1 else tokenize(rec1.get("raw_name", "")))
    toks2 = rec2.get("name_tokens") if "name_tokens" in rec2 else (tokenize(n2) if n2 else tokenize(rec2.get("raw_name", "")))

    ngs1 = rec1.get("name_ngrams") if "name_ngrams" in rec1 else char_ngrams(n1, 3)
    ngs2 = rec2.get("name_ngrams") if "name_ngrams" in rec2 else char_ngrams(n2, 3)

    addr_toks1 = rec1.get("addr_tokens") if "addr_tokens" in rec1 else tokenize(rec1.get("raw_addr", ""))
    addr_toks2 = rec2.get("addr_tokens") if "addr_tokens" in rec2 else tokenize(rec2.get("raw_addr", ""))

    digs1 = rec1.get("digits") if "digits" in rec1 else extract_digit_blocks(rec1.get("raw_addr", ""))
    digs2 = rec2.get("digits") if "digits" in rec2 else extract_digit_blocks(rec2.get("raw_addr", ""))

    addr1_empty = rec1.get("is_addr_empty", not bool(rec1.get("raw_addr", "").strip()))
    addr2_empty = rec2.get("is_addr_empty", not bool(rec2.get("raw_addr", "").strip()))

    # 1. name_exact
    f_name_exact = 1.0 if (n1 and n1 == n2) else 0.0

    # 2. name_tok_jaccard
    f_name_tok_jaccard = token_jaccard(toks1, toks2)

    # 3. name_tok_overlap
    f_name_tok_overlap = token_overlap(toks1, toks2)

    # 4. name_tok_containment
    f_name_tok_containment = token_containment(toks1, toks2)

    # 5. name_ngram_jaccard
    f_name_ngram_jaccard = token_jaccard(ngs1, ngs2)

    # 6. name_levenshtein_ratio
    f_name_levenshtein_ratio = levenshtein_ratio(n1, n2)

    # 7. name_first_tok_match
    f1 = rec1.get("first_tok") or first_token(n1)
    f2 = rec2.get("first_tok") or first_token(n2)
    f_name_first_tok_match = 1.0 if (f1 and f1 == f2) else 0.0

    # 8. name_len_diff_ratio
    max_len = max(len(n1), len(n2))
    f_name_len_diff_ratio = float(abs(len(n1) - len(n2)) / max_len) if max_len > 0 else 0.0

    # 9. addr_tok_jaccard
    f_addr_tok_jaccard = token_jaccard(addr_toks1, addr_toks2)

    # 10. addr_tok_overlap
    f_addr_tok_overlap = token_overlap(addr_toks1, addr_toks2)

    # 11. addr_tok_containment
    f_addr_tok_containment = token_containment(addr_toks1, addr_toks2)

    # 12. addr_missing_either
    f_addr_missing_either = 1.0 if (addr1_empty or addr2_empty) else 0.0

    # 13. shared_digit_count
    shared_digs = digs1 & digs2
    f_shared_digit_count = float(len(shared_digs))

    # 14. has_shared_digits
    f_has_shared_digits = 1.0 if bool(shared_digs) else 0.0

    # 15. conflicting_digits
    f_conflicting_digits = 1.0 if (digs1 and digs2 and not shared_digs) else 0.0

    # 16. domain_stem_match
    dom_stem1 = rec1.get("domain_stem") if "domain_stem" in rec1 else extract_domain_stem(rec1.get("raw_name", ""))
    dom_stem2 = rec2.get("domain_stem") if "domain_stem" in rec2 else extract_domain_stem(rec2.get("raw_name", ""))
    stripped1 = rec1.get("stripped_name") if "stripped_name" in rec1 else re.sub(r'[^a-z0-9]', '', n1)
    stripped2 = rec2.get("stripped_name") if "stripped_name" in rec2 else re.sub(r'[^a-z0-9]', '', n2)

    dom_match = False
    if dom_stem2 and len(dom_stem2) >= 3:
        if dom_stem2 == stripped1 or dom_stem2 in toks1:
            dom_match = True
    if not dom_match and dom_stem1 and len(dom_stem1) >= 3:
        if dom_stem1 == stripped2 or dom_stem1 in toks2:
            dom_match = True
    f_domain_stem_match = 1.0 if dom_match else 0.0

    # 17. is_source3
    f_is_source3 = 1.0 if is_source3 else 0.0

    # 18. cand_pool_size
    f_cand_pool_size = float(cand_pool_size)

    return [
        f_name_exact,
        f_name_tok_jaccard,
        f_name_tok_overlap,
        f_name_tok_containment,
        f_name_ngram_jaccard,
        f_name_levenshtein_ratio,
        f_name_first_tok_match,
        f_name_len_diff_ratio,
        f_addr_tok_jaccard,
        f_addr_tok_overlap,
        f_addr_tok_containment,
        f_addr_missing_either,
        f_shared_digit_count,
        f_has_shared_digits,
        f_conflicting_digits,
        f_domain_stem_match,
        f_is_source3,
        f_cand_pool_size,
    ]


def extract_feature_dict(
    rec1: Dict[str, Any],
    rec2: Dict[str, Any],
    is_source3: bool = False,
    cand_pool_size: int = 1
) -> Dict[str, float]:
    """Helper to return features as a dictionary keyed by feature name."""
    vec = extract_feature_vector(rec1, rec2, is_source3, cand_pool_size)
    return dict(zip(FEATURE_NAMES, vec))
