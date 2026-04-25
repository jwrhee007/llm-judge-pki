#!/usr/bin/env python3
"""Exp.2-3 Prompt Mitigation 3-way analysis (Standard / Direct / CoT).

Re-runs the analyses produced ad-hoc during Exp.2-3 (basic metrics, ΔPKI,
Knowledge-Probe breakdown, Standard PKI overlap, Direct vs CoT 4-quadrant,
CoT reasoning markers, sanity check) and saves a structured summary.

Inputs (read-only):
  results/context_swap/same_swap_results.jsonl         (Standard, Exp.2-2)
  results/prompt_mitigation/P-Lee-Direct/same_swap_results.jsonl
  results/prompt_mitigation/P-Lee-CoT/same_swap_results.jsonl
  results/probe/probe_method_A.jsonl                   (Knowledge Probe)
  data/batch_outputs/exp{2_2,23_*}_same_swap_chunk*_output.jsonl  (sanity)

Output:
  results/prompt_mitigation/analysis_summary.json

Usage:
  python scripts/07_analyze_prompt_mitigation.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPTS = ["Standard", "Direct", "CoT"]
RESULT_PATHS = {
    "Standard": PROJECT_ROOT / "results/context_swap/same_swap_results.jsonl",
    "Direct":   PROJECT_ROOT / "results/prompt_mitigation/P-Lee-Direct/same_swap_results.jsonl",
    "CoT":      PROJECT_ROOT / "results/prompt_mitigation/P-Lee-CoT/same_swap_results.jsonl",
}
BATCH_GLOB = {
    "Standard": "exp2_2_same_swap_chunk*_output.jsonl",
    "Direct":   "exp23_P-Lee-Direct_same_swap_chunk*_output.jsonl",
    "CoT":      "exp23_P-Lee-CoT_same_swap_chunk*_output.jsonl",
}
PROBE_PATH = PROJECT_ROOT / "results/probe/probe_method_A.jsonl"
BATCH_DIR = PROJECT_ROOT / "data/batch_outputs"
SUMMARY_OUT = PROJECT_ROOT / "results/prompt_mitigation/analysis_summary.json"

# Reasoning markers — literal substrings (lowercased), matches OR-style within group.
# Mirrors the four marker themes reported in the Exp.2-3 narrative analysis.
REASONING_MARKERS = {
    "factually_correct": [
        "factually correct",
        "based on general knowledge",
    ],
    "does_not_contradict": [
        "does not contradict",
        "no contradiction",   # also matches "no contradictions"
    ],
    "external_knowledge": [
        "general knowledge",
        "historical fact",
        "well-known",
    ],
    "adversarial_pivot": [
        "however",
        "but is still",
    ],
}

# Short aliases used in the per-qid matrix (for compact JSON output).
MARKER_ALIASES = {
    "factually_correct":   "FC",
    "does_not_contradict": "DNC",
    "external_knowledge":  "EK",
    "adversarial_pivot":   "AP",
}

KNOWLEDGE_LEVELS = ["strong-knows", "weak-knows", "guess", "doesn't-know"]


# ---------- Loading ----------

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_knowledge_probe(method="A"):
    """Return {qid: knowledge_level}."""
    path = PROJECT_ROOT / f"results/probe/probe_method_{method}.jsonl"
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["question_id"]] = r["knowledge_level"]
    return out


# ---------- Basic metrics ----------

def compute_basic_metrics(items):
    n = len(items)
    if n == 0:
        return {"n": 0}
    dist = Counter(it["majority_verdict"] for it in items)
    entropies = [it["verdict_entropy"] for it in items]
    return {
        "n": n,
        "PKI_count": dist.get("CORRECT", 0),
        "CAR_count": dist.get("NOT_ATTEMPTED", 0),
        "ORR_count": dist.get("INCORRECT", 0),
        "PKI": dist.get("CORRECT", 0) / n,
        "CAR": dist.get("NOT_ATTEMPTED", 0) / n,
        "ORR": dist.get("INCORRECT", 0) / n,
        "mean_entropy": float(np.mean(entropies)),
        "unstable_count": int(sum(1 for e in entropies if e > 0)),
    }


# ---------- Overlap ----------

def compute_overlap_analysis(std_items, mit_items):
    """Std PKI overlap with a single mitigation prompt."""
    std_q = {it["question_id"]: it for it in std_items}
    mit_q = {it["question_id"]: it for it in mit_items}
    common = set(std_q) & set(mit_q)

    std_pki = {q for q in common if std_q[q]["majority_verdict"] == "CORRECT"}
    mit_pki = {q for q in common if mit_q[q]["majority_verdict"] == "CORRECT"}

    resolved = std_pki - mit_pki
    unresolved = std_pki & mit_pki
    new_pki = mit_pki - std_pki

    det_std = {
        q for q in std_pki
        if std_q[q]["verdict_distribution"].get("CORRECT", 0) == 30
    }
    det_resolved = {q for q in det_std if mit_q[q]["majority_verdict"] != "CORRECT"}

    return {
        "common_qids": len(common),
        "std_pki_n": len(std_pki),
        "mit_pki_n": len(mit_pki),
        "resolved": len(resolved),
        "unresolved": len(unresolved),
        "new_pki": len(new_pki),
        "deterministic_std_pki_n": len(det_std),
        "deterministic_resolved": len(det_resolved),
        "resolved_qids": sorted(resolved),
        "unresolved_qids": sorted(unresolved),
        "new_pki_qids": sorted(new_pki),
        "deterministic_std_pki_qids": sorted(det_std),
    }


# ---------- 4-Quadrant ----------

def compute_quadrant(std_items, direct_items, cot_items):
    """Direct vs CoT outcome on each Standard PKI item."""
    std_q = {it["question_id"]: it for it in std_items}
    d_q = {it["question_id"]: it for it in direct_items}
    c_q = {it["question_id"]: it for it in cot_items}
    common = set(std_q) & set(d_q) & set(c_q)

    std_pki = {q for q in common if std_q[q]["majority_verdict"] == "CORRECT"}
    # Restrict to std_pki: Direct/CoT still PKI on these items
    d_unresolved = {q for q in std_pki if d_q[q]["majority_verdict"] == "CORRECT"}
    c_unresolved = {q for q in std_pki if c_q[q]["majority_verdict"] == "CORRECT"}

    both_resolved = std_pki - d_unresolved - c_unresolved
    direct_only_resolved = c_unresolved - d_unresolved
    cot_only_resolved = d_unresolved - c_unresolved
    both_failed = d_unresolved & c_unresolved

    return {
        "common_qids": len(common),
        "std_pki_n": len(std_pki),
        "both_resolved": len(both_resolved),
        "direct_only_resolved": len(direct_only_resolved),
        "cot_only_resolved": len(cot_only_resolved),
        "both_failed": len(both_failed),
        "both_resolved_qids": sorted(both_resolved),
        "direct_only_resolved_qids": sorted(direct_only_resolved),
        "cot_only_resolved_qids": sorted(cot_only_resolved),
        "both_failed_qids": sorted(both_failed),
    }


# ---------- Probe breakdown ----------

def compute_probe_breakdown(items_by_prompt, probe, common_with_probe):
    """For each (prompt, level), basic metrics restricted to that level."""
    out = {p: {} for p in PROMPTS}
    for p in PROMPTS:
        q_to_item = {it["question_id"]: it for it in items_by_prompt[p]}
        for lvl in KNOWLEDGE_LEVELS:
            level_items = [q_to_item[q] for q in common_with_probe if probe.get(q) == lvl]
            out[p][lvl] = compute_basic_metrics(level_items)
    return out


# ---------- Reasoning markers ----------

def parse_first_correct_trial(item):
    """Return response text of first trial with verdict=='CORRECT', or None."""
    for t in item["trials"]:
        if t.get("verdict") == "CORRECT":
            return t.get("response")
    return None


def analyze_reasoning_markers(cot_items, slice_limit=10):
    """Extract first-CORRECT-trial reasoning from CoT majority=CORRECT items
    and count marker hits. Slice to first `slice_limit` items to match the
    narrative report; full count is also returned."""
    pki_items = [it for it in cot_items if it["majority_verdict"] == "CORRECT"]
    sliced = pki_items[:slice_limit]

    extracted = []
    for it in sliced:
        resp = parse_first_correct_trial(it)
        extracted.append({
            "question_id": it["question_id"],
            "ner_tag": it.get("ner_tag"),
            "correct_count": it["verdict_distribution"].get("CORRECT", 0),
            "response": resp or "",
        })

    n_extracted = sum(1 for e in extracted if e["response"])
    counts = {name: 0 for name in REASONING_MARKERS}
    matrix = {}  # qid → {alias: bool} per-trace marker hits
    for e in extracted:
        qid = e["question_id"]
        if not e["response"]:
            matrix[qid] = None
            continue
        text = e["response"].lower()
        hits = {}
        for name, keywords in REASONING_MARKERS.items():
            hit = any(k.lower() in text for k in keywords)
            hits[MARKER_ALIASES[name]] = hit
            if hit:
                counts[name] += 1
        matrix[qid] = hits

    return {
        "n_total_pki_items": len(pki_items),
        "n_extracted": n_extracted,
        "slice_limit": slice_limit,
        "marker_counts": counts,
        "marker_matrix": matrix,
        "samples": extracted,
    }


# ---------- Sanity check ----------

def check_sanity(prompt_id):
    """Aggregate finish_reason + content length across raw batch outputs."""
    pattern = BATCH_GLOB[prompt_id]
    files = sorted(BATCH_DIR.glob(pattern))
    if not files:
        return None

    fr = Counter()
    lengths = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                choices = r.get("response", {}).get("body", {}).get("choices", [{}])
                fr[choices[0].get("finish_reason")] += 1
                content = choices[0].get("message", {}).get("content", "")
                lengths.append(len(content))

    return {
        "n_chunks": len(files),
        "total": sum(fr.values()),
        "finish_reason": dict(fr),
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "length_mean": float(np.mean(lengths)) if lengths else None,
    }


# ---------- Pretty printing ----------

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def render_basic_table(metrics_by_prompt):
    rows = []
    for p in PROMPTS:
        m = metrics_by_prompt[p]
        rows.append([
            p,
            m["n"],
            f"{m['PKI']*100:.2f}% ({m['PKI_count']}/{m['n']})",
            f"{m['CAR']*100:.2f}% ({m['CAR_count']}/{m['n']})",
            f"{m['ORR']*100:.2f}% ({m['ORR_count']}/{m['n']})",
            f"{m['mean_entropy']:.4f}",
            f"{m['unstable_count']} ({m['unstable_count']/m['n']*100:.1f}%)",
        ])
    print(tabulate(
        rows,
        headers=["Prompt", "N", "PKI", "CAR", "ORR", "MeanH", "Unstable"],
        tablefmt="github",
    ))


def render_delta_table(metrics_by_prompt):
    base = metrics_by_prompt["Standard"]["PKI"]
    rows = []
    for p in ["Direct", "CoT"]:
        m = metrics_by_prompt[p]["PKI"]
        delta_abs = m - base
        delta_rel = (m - base) / base * 100 if base else 0.0
        rows.append([
            p,
            f"{m*100:.2f}%",
            f"{base*100:.2f}%",
            f"{delta_abs*100:+.2f} pp",
            f"{delta_rel:+.1f}%",
        ])
    print(tabulate(
        rows,
        headers=["Prompt", "PKI", "Std baseline", "ΔPKI (abs)", "Δ (rel)"],
        tablefmt="github",
    ))


def render_probe_table(by_prompt):
    rows = []
    # n is identical across prompts (shared common_with_probe set)
    for lvl in KNOWLEDGE_LEVELS:
        n = by_prompt["Standard"][lvl]["n"]
        rows.append([
            lvl, n,
            f"{by_prompt['Standard'][lvl]['PKI']*100:.2f}%",
            f"{by_prompt['Direct'][lvl]['PKI']*100:.2f}%",
            f"{by_prompt['CoT'][lvl]['PKI']*100:.2f}%",
        ])
    print(tabulate(
        rows,
        headers=["Level", "n", "Std PKI", "Direct PKI", "CoT PKI"],
        tablefmt="github",
    ))


# ---------- Main ----------

def main():
    print_section("Exp.2-3 Prompt Mitigation Analysis (Standard / Direct / CoT)")

    items_by_prompt = {p: load_results(RESULT_PATHS[p]) for p in PROMPTS}
    probe = load_knowledge_probe("A")

    # 1. Basic metrics
    metrics = {p: compute_basic_metrics(items_by_prompt[p]) for p in PROMPTS}
    print_section("1. Basic Metrics (item-level, majority verdict)")
    render_basic_table(metrics)

    # 2. ΔPKI
    print_section("2. ΔPKI vs Standard baseline")
    render_delta_table(metrics)

    # 3. Probe breakdown
    common_3way = set.intersection(*[
        {it["question_id"] for it in items_by_prompt[p]} for p in PROMPTS
    ])
    common_with_probe = common_3way & set(probe)
    probe_breakdown = compute_probe_breakdown(items_by_prompt, probe, common_with_probe)
    print_section("3. Knowledge Probe (Method A) breakdown — PKI by level")
    print(f"  common qids (3-way ∩) = {len(common_3way)},  with Probe = {len(common_with_probe)}")
    print()
    render_probe_table(probe_breakdown)

    # 4. Overlap (Std PKI ∩ mitigation)
    print_section("4. Overlap Analysis (Standard PKI ∩ mitigation)")
    overlap = {}
    for p in ["Direct", "CoT"]:
        ov = compute_overlap_analysis(items_by_prompt["Standard"], items_by_prompt[p])
        overlap[p] = ov
        print(f"\n  --- {p} ---")
        print(f"  common qids: {ov['common_qids']}  |  Std PKI: {ov['std_pki_n']}, {p} PKI: {ov['mit_pki_n']}")
        print(f"  Resolved   : {ov['resolved']}/{ov['std_pki_n']} "
              f"({ov['resolved']/max(ov['std_pki_n'],1)*100:.1f}%)")
        print(f"  Unresolved : {ov['unresolved']}")
        print(f"  New PKI    : {ov['new_pki']}")
        print(f"  Deterministic Std PKI (30/30): {ov['deterministic_std_pki_n']}, "
              f"resolved {ov['deterministic_resolved']} "
              f"({ov['deterministic_resolved']/max(ov['deterministic_std_pki_n'],1)*100:.1f}%)")

    # 5. 4-Quadrant
    print_section("5. Direct vs CoT 4-Quadrant on Standard PKI items")
    quad = compute_quadrant(
        items_by_prompt["Standard"],
        items_by_prompt["Direct"],
        items_by_prompt["CoT"],
    )
    print(f"  common qids (3-way): {quad['common_qids']}  |  Std PKI base: {quad['std_pki_n']}")
    print()
    rows = [
        ["Both resolved",        quad["both_resolved"],
         ", ".join(quad["both_resolved_qids"][:6]) +
         (" ..." if quad["both_resolved"] > 6 else "")],
        ["Direct only resolved", quad["direct_only_resolved"],
         ", ".join(quad["direct_only_resolved_qids"]) or "—"],
        ["CoT only resolved",    quad["cot_only_resolved"],
         ", ".join(quad["cot_only_resolved_qids"]) or "—"],
        ["Both failed",          quad["both_failed"],
         ", ".join(quad["both_failed_qids"]) or "—"],
    ]
    print(tabulate(rows, headers=["Quadrant", "n", "qids"], tablefmt="github"))

    # 6. CoT reasoning markers
    print_section("6. CoT Reasoning Marker Analysis (PKI items, first CORRECT trial)")
    markers = analyze_reasoning_markers(items_by_prompt["CoT"])
    print(f"  CoT majority=CORRECT items: {markers['n_total_pki_items']}")
    print(f"  Slice limit (matches report): {markers['slice_limit']}")
    print(f"  Extracted (response present): {markers['n_extracted']}")
    print()
    denom = min(markers["n_extracted"], markers["slice_limit"])
    rows = []
    for name, count in markers["marker_counts"].items():
        rows.append([name, f"{count}/{denom}",
                     f"{count/max(denom,1)*100:.0f}%"])
    print(tabulate(rows, headers=["Marker", "Hits", "Rate"], tablefmt="github"))
    print("\n  Representative samples (first 3 traces):")
    for s in markers["samples"][:3]:
        snippet = s["response"][:200] + (" ..." if len(s["response"]) > 200 else "")
        print(f"    --- qid={s['question_id']}  ner={s['ner_tag']}  "
              f"CORRECT/30={s['correct_count']} ---")
        print(f"    {snippet}")

    # 7. Sanity check
    print_section("7. Sanity Check (raw batch_outputs)")
    sanity = {}
    rows = []
    for p in PROMPTS:
        s = check_sanity(p)
        sanity[p] = s
        if s is None:
            rows.append([p, "—", "—", "—", "—", "—"])
            continue
        rows.append([
            p, s["n_chunks"], s["total"],
            ", ".join(f"{k}={v}" for k, v in s["finish_reason"].items()),
            f"{s['length_min']}-{s['length_max']}",
            f"{s['length_mean']:.0f}" if s["length_mean"] is not None else "—",
        ])
    print(tabulate(
        rows,
        headers=["Prompt", "Chunks", "Total", "finish_reason", "len min-max", "len mean"],
        tablefmt="github",
    ))

    # 8. Save summary
    out = {
        "metrics": metrics,
        "delta_pki": {
            p: {
                "PKI": metrics[p]["PKI"],
                "delta_abs_pp": (metrics[p]["PKI"] - metrics["Standard"]["PKI"]) * 100,
                "delta_rel_pct": ((metrics[p]["PKI"] - metrics["Standard"]["PKI"])
                                  / metrics["Standard"]["PKI"] * 100
                                  if metrics["Standard"]["PKI"] else 0.0),
            }
            for p in ["Direct", "CoT"]
        },
        "common_qids_3way": len(common_3way),
        "common_qids_with_probe": len(common_with_probe),
        "probe_breakdown": probe_breakdown,
        "overlap": overlap,
        "quadrant": quad,
        "reasoning_markers": {
            "n_total_pki_items": markers["n_total_pki_items"],
            "n_extracted": markers["n_extracted"],
            "slice_limit": markers["slice_limit"],
            "marker_counts": markers["marker_counts"],
            "marker_alias_legend": {v: k for k, v in MARKER_ALIASES.items()},
            "marker_matrix": markers["marker_matrix"],
            "sample_qids": [s["question_id"] for s in markers["samples"]],
        },
        "sanity": sanity,
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print()
    print(f"Saved summary → {SUMMARY_OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
