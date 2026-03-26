#!/usr/bin/env python3
"""
Step 3: Knowledge Probe 분석 리포트 생성.

방식 A / B 결과를 비교 분석하고 아래 내용을 포함하는 리포트를 생성한다:
  1. 전체 4단계 분류 분포 (방식 A vs B)
  2. NER 태그별 분류 분포
  3. 방식 A/B 간 일치도 (Cohen's Kappa)
  4. 방식 A/B 간 교차 테이블 (Confusion Matrix)
  5. 시각화 (bar chart, heatmap)

채택 기준: PKI rate와의 상관이 더 높은 방식을 채택한다.
(PKI rate는 Exp. 2-2 이후에 확정되므로, 이 단계에서는 분포 비교만 수행)

Usage:
    python scripts/03_analyze_probe.py
    python scripts/03_analyze_probe.py --method-a results/probe/probe_method_A.jsonl \
                                       --method-b results/probe/probe_method_B.jsonl
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probes.knowledge_probe import compute_probe_summary, load_probe_results
from src.utils.logger import setup_logger

logger = setup_logger("analyze_probe")

KNOWLEDGE_LEVELS = ["strong-knows", "weak-knows", "guess", "doesn't-know"]
LEVEL_ORDER = {level: i for i, level in enumerate(KNOWLEDGE_LEVELS)}


# =====================================================================
# Analysis functions
# =====================================================================

def build_comparison_df(
    results_a: list[dict],
    results_b: list[dict],
) -> pd.DataFrame:
    """
    방식 A/B 결과를 question_id 기준으로 병합한 DataFrame 생성.
    """
    a_map = {r["question_id"]: r for r in results_a}
    b_map = {r["question_id"]: r for r in results_b}

    common_ids = sorted(set(a_map.keys()) & set(b_map.keys()))
    if len(common_ids) < len(a_map):
        logger.warning(
            f"Mismatched question IDs: A={len(a_map)}, B={len(b_map)}, "
            f"common={len(common_ids)}"
        )

    rows = []
    for qid in common_ids:
        ra, rb = a_map[qid], b_map[qid]
        rows.append({
            "question_id": qid,
            "question": ra["question"],
            "gold_answer": ra["gold_answer"],
            "ner_tag": ra.get("ner_tag", "UNKNOWN"),
            "level_A": ra["knowledge_level"],
            "n_correct_A": ra["n_correct"],
            "level_B": rb["knowledge_level"],
            "n_correct_B": rb["n_correct"],
            "agreement": ra["knowledge_level"] == rb["knowledge_level"],
        })

    return pd.DataFrame(rows)


def compute_cohens_kappa(df: pd.DataFrame) -> float:
    """
    방식 A/B 간 Cohen's Kappa 계산.
    """
    labels = KNOWLEDGE_LEVELS
    n = len(df)
    if n == 0:
        return 0.0

    # Confusion matrix
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for _, row in df.iterrows():
        i = LEVEL_ORDER.get(row["level_A"], 0)
        j = LEVEL_ORDER.get(row["level_B"], 0)
        matrix[i, j] += 1

    # Observed agreement
    p_o = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    p_e = (row_sums * col_sums).sum() / (n * n)

    if p_e == 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa


def build_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    방식 A (rows) vs 방식 B (columns) 교차 테이블.
    """
    matrix = pd.crosstab(
        df["level_A"],
        df["level_B"],
        margins=True,
        margins_name="Total",
    )
    # Reorder
    order = [l for l in KNOWLEDGE_LEVELS if l in matrix.index] + ["Total"]
    cols = [l for l in KNOWLEDGE_LEVELS if l in matrix.columns] + ["Total"]
    matrix = matrix.reindex(index=order, columns=cols, fill_value=0)
    return matrix


def build_ner_tag_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    NER 태그별 분류 분포 (방식 A/B 나란히).
    """
    rows = []
    for tag in sorted(df["ner_tag"].unique()):
        sub = df[df["ner_tag"] == tag]
        n = len(sub)
        row = {"ner_tag": tag, "n": n}
        for method_col, prefix in [("level_A", "A"), ("level_B", "B")]:
            counts = sub[method_col].value_counts()
            for level in KNOWLEDGE_LEVELS:
                count = counts.get(level, 0)
                row[f"{prefix}_{level}"] = count
                row[f"{prefix}_{level}_pct"] = count / n * 100 if n > 0 else 0
        rows.append(row)

    return pd.DataFrame(rows)


def generate_visualizations(
    df: pd.DataFrame,
    summary_a: dict,
    summary_b: dict,
    output_dir: str,
) -> None:
    """
    시각화 생성 (matplotlib).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping visualizations.")
        return

    output = Path(output_dir)

    # --- 1. Overall distribution comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (method_label, summary) in zip(
        axes, [("Method A (Bare)", summary_a), ("Method B (Eliciting)", summary_b)]
    ):
        overall = summary["overall"]
        counts = [overall.get(level, 0) for level in KNOWLEDGE_LEVELS]
        total = sum(counts)
        pcts = [c / total * 100 if total > 0 else 0 for c in counts]
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

        bars = ax.bar(KNOWLEDGE_LEVELS, pcts, color=colors, edgecolor="white")
        ax.set_title(method_label, fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, max(pcts) * 1.3 if pcts else 100)

        for bar, pct, cnt in zip(bars, pcts, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{pct:.1f}%\n(n={cnt})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle("Knowledge Probe: 4-Level Classification", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output / "probe_overall_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output / 'probe_overall_comparison.png'}")

    # --- 2. Confusion matrix heatmap ---
    cm = build_confusion_matrix(df)
    cm_no_total = cm.iloc[:-1, :-1]  # Remove margins for heatmap

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_no_total,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Method B", fontsize=12)
    ax.set_ylabel("Method A", fontsize=12)
    ax.set_title("Method A vs B — Cross-tabulation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output / "probe_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output / 'probe_confusion_matrix.png'}")

    # --- 3. NER-tag breakdown ---
    ner_table = build_ner_tag_table(df)
    tags = ner_table["ner_tag"].tolist()
    n_tags = len(tags)
    if n_tags == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, n_tags * 0.5)))

    for ax, prefix, title in [
        (axes[0], "A", "Method A (Bare)"),
        (axes[1], "B", "Method B (Eliciting)"),
    ]:
        bottoms = np.zeros(n_tags)
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        for level, color in zip(KNOWLEDGE_LEVELS, colors):
            vals = ner_table[f"{prefix}_{level}_pct"].values
            ax.barh(tags, vals, left=bottoms, color=color, label=level, height=0.6)
            bottoms += vals

        ax.set_xlabel("Percentage (%)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.invert_yaxis()

    axes[1].legend(
        loc="lower right", fontsize=8, title="Knowledge Level", title_fontsize=9
    )
    plt.suptitle("Knowledge Probe by NER Tag", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output / "probe_ner_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output / 'probe_ner_breakdown.png'}")


# =====================================================================
# Report generation
# =====================================================================

def generate_report(
    df: pd.DataFrame,
    summary_a: dict,
    summary_b: dict,
    output_path: str,
) -> None:
    """Markdown 형식의 분석 리포트를 생성한다."""
    lines = []
    lines.append("# Knowledge Probe Analysis Report (Exp. 2-0)")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"- Total items: **{len(df)}**")
    lines.append(f"- Method A (Bare question): {summary_a['total']} items")
    lines.append(f"- Method B (Knowledge-eliciting): {summary_b['total']} items")
    lines.append("")

    # Overall distribution
    lines.append("## 2. Overall 4-Level Classification")
    lines.append("")
    lines.append("| Level | Method A | A (%) | Method B | B (%) |")
    lines.append("|-------|----------|-------|----------|-------|")
    for level in KNOWLEDGE_LEVELS:
        ca = summary_a["overall"].get(level, 0)
        cb = summary_b["overall"].get(level, 0)
        pa = ca / summary_a["total"] * 100 if summary_a["total"] > 0 else 0
        pb = cb / summary_b["total"] * 100 if summary_b["total"] > 0 else 0
        lines.append(f"| {level} | {ca} | {pa:.1f}% | {cb} | {pb:.1f}% |")
    lines.append("")

    # Agreement
    kappa = compute_cohens_kappa(df)
    agreement_rate = df["agreement"].mean() * 100
    lines.append("## 3. Method A vs B Agreement")
    lines.append("")
    lines.append(f"- **Exact agreement rate**: {agreement_rate:.1f}%")
    lines.append(f"- **Cohen's Kappa**: {kappa:.3f}")
    lines.append("")

    # Kappa interpretation
    if kappa >= 0.81:
        interp = "Almost perfect agreement"
    elif kappa >= 0.61:
        interp = "Substantial agreement"
    elif kappa >= 0.41:
        interp = "Moderate agreement"
    elif kappa >= 0.21:
        interp = "Fair agreement"
    else:
        interp = "Slight or poor agreement"
    lines.append(f"- **Interpretation**: {interp}")
    lines.append("")

    # Confusion matrix
    lines.append("## 4. Cross-tabulation (Method A × Method B)")
    lines.append("")
    cm = build_confusion_matrix(df)
    lines.append(cm.to_markdown())
    lines.append("")

    # Disagreement analysis
    disagree = df[~df["agreement"]]
    if len(disagree) > 0:
        lines.append("## 5. Disagreement Analysis")
        lines.append("")
        lines.append(f"Total disagreements: {len(disagree)}")
        lines.append("")

        # Most common transitions
        transitions = Counter()
        for _, row in disagree.iterrows():
            transitions[(row["level_A"], row["level_B"])] += 1

        lines.append("| Method A → Method B | Count |")
        lines.append("|---------------------|-------|")
        for (la, lb), cnt in transitions.most_common(10):
            lines.append(f"| {la} → {lb} | {cnt} |")
        lines.append("")

    # NER tag breakdown
    lines.append("## 6. NER Tag Breakdown")
    lines.append("")
    ner_table = build_ner_tag_table(df)
    # Simplified view
    display_cols = ["ner_tag", "n"]
    for prefix in ["A", "B"]:
        for level in KNOWLEDGE_LEVELS:
            display_cols.append(f"{prefix}_{level}")

    lines.append(ner_table[display_cols].to_markdown(index=False))
    lines.append("")

    # knows rate comparison
    lines.append("## 7. Knows Rate Comparison")
    lines.append("")
    lines.append("*Knows rate = (strong-knows + weak-knows) / total*")
    lines.append("")

    knows_a = (
        summary_a["overall"].get("strong-knows", 0)
        + summary_a["overall"].get("weak-knows", 0)
    )
    knows_b = (
        summary_b["overall"].get("strong-knows", 0)
        + summary_b["overall"].get("weak-knows", 0)
    )
    total = summary_a["total"]
    kr_a = knows_a / total * 100 if total > 0 else 0
    kr_b = knows_b / total * 100 if total > 0 else 0

    lines.append(f"- Method A knows rate: **{kr_a:.1f}%** ({knows_a}/{total})")
    lines.append(f"- Method B knows rate: **{kr_b:.1f}%** ({knows_b}/{total})")
    lines.append("")
    if kr_b > kr_a:
        lines.append(
            f"Method B는 Method A보다 knows rate이 {kr_b - kr_a:.1f}pp 높다. "
            "Knowledge-eliciting prompt가 모델의 지식 recall을 더 잘 유도함."
        )
    elif kr_a > kr_b:
        lines.append(
            f"Method A는 Method B보다 knows rate이 {kr_a - kr_b:.1f}pp 높다."
        )
    else:
        lines.append("두 방식의 knows rate이 동일하다.")
    lines.append("")

    # Next steps
    lines.append("## 8. Next Steps")
    lines.append("")
    lines.append(
        "- 방식 채택 기준: Exp. 2-2 Context-Swap 실행 후 "
        "PKI rate와의 상관이 더 높은 방식을 최종 채택"
    )
    lines.append(
        "- 채택하지 않은 방식의 결과는 appendix에 비교 테이블로 보고"
    )
    lines.append(
        "- H-KNOW 가설 검증: "
        "PKI rate가 knowledge probe 단계에 따라 단조 증가하는지 확인 "
        "(strong-knows > weak-knows > guess > doesn't-know)"
    )
    lines.append("")

    # Write
    report_text = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Report saved to {output_path}")


def generate_disagreement_analysis(
    results_a: list[dict],
    results_b: list[dict],
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    A/B 불일치 문항의 상세 응답을 텍스트 파일로 저장한다.
    """
    disagree = df[~df["agreement"]]
    if len(disagree) == 0:
        logger.info("No disagreements found. Skipping disagreement analysis.")
        return

    # question_id → result 매핑
    a_map = {r["question_id"]: r for r in results_a}
    b_map = {r["question_id"]: r for r in results_b}

    disagree_ids = set(disagree["question_id"].tolist())

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Disagreement Analysis: {len(disagree_ids)} items\n")
        out.write(f"Generated at: {pd.Timestamp.now().isoformat()}\n")
        out.write("=" * 80 + "\n\n")

        for qid in sorted(disagree_ids):
            a = a_map.get(qid)
            b = b_map.get(qid)
            if not a or not b:
                continue

            out.write("=" * 80 + "\n")
            out.write(f"Q: {a['question']}\n")
            out.write(f"Gold: {a['gold_answer']}\n")
            out.write(f"NER: {a.get('ner_tag', 'UNKNOWN')}\n")
            out.write(
                f"A: {a['knowledge_level']} ({a['n_correct']}/3)  |  "
                f"B: {b['knowledge_level']} ({b['n_correct']}/3)\n\n"
            )

            for label, item in [("A", a), ("B", b)]:
                for t in item["trials"]:
                    correct_mark = "O" if t["is_correct"] else "X"
                    resp = t["response"] if t["response"] else "None"
                    out.write(
                        f"  [{label}] Trial {t['trial_idx']} ({correct_mark}):\n"
                        f"    {resp}\n\n"
                    )
                out.write("\n")

    logger.info(
        f"Disagreement analysis saved to {output_path} "
        f"({len(disagree_ids)} items)"
    )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze Knowledge Probe results")
    parser.add_argument(
        "--method-a",
        default="results/probe/probe_method_A.jsonl",
        help="Method A results file",
    )
    parser.add_argument(
        "--method-b",
        default="results/probe/probe_method_B.jsonl",
        help="Method B results file",
    )
    parser.add_argument(
        "--output-dir",
        default="results/probe",
        help="Output directory for report and figures",
    )
    parser.add_argument(
        "--single-method",
        choices=["A", "B"],
        default=None,
        help="Analyze a single method only (skip comparison)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 3: Knowledge Probe Analysis")
    logger.info("=" * 60)

    if args.single_method:
        # Single method analysis
        method = args.single_method
        path = args.method_a if method == "A" else args.method_b
        results = load_probe_results(path)
        summary = compute_probe_summary(results)

        logger.info(f"\nMethod {method} Summary:")
        for level in KNOWLEDGE_LEVELS:
            count = summary["overall"].get(level, 0)
            pct = count / summary["total"] * 100 if summary["total"] > 0 else 0
            logger.info(f"  {level:15s}: {count:4d}  ({pct:5.1f}%)")

        summary_path = output_dir / f"probe_method_{method}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")
        return

    # Dual method comparison
    if not Path(args.method_a).exists():
        logger.error(f"Method A results not found: {args.method_a}")
        sys.exit(1)
    if not Path(args.method_b).exists():
        logger.error(f"Method B results not found: {args.method_b}")
        sys.exit(1)

    results_a = load_probe_results(args.method_a)
    results_b = load_probe_results(args.method_b)

    summary_a = compute_probe_summary(results_a)
    summary_b = compute_probe_summary(results_b)

    # Build comparison DataFrame
    df = build_comparison_df(results_a, results_b)
    logger.info(f"Comparison DataFrame: {len(df)} rows")

    # Save comparison CSV
    csv_path = output_dir / "probe_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison CSV saved to {csv_path}")

    # Generate report
    report_path = output_dir / "probe_analysis_report.md"
    generate_report(df, summary_a, summary_b, str(report_path))

    # Generate disagreement analysis
    disagree_path = output_dir / "disagreement_analysis.txt"
    generate_disagreement_analysis(results_a, results_b, df, str(disagree_path))

    # Generate visualizations
    generate_visualizations(df, summary_a, summary_b, str(output_dir))

    # Print key metrics
    kappa = compute_cohens_kappa(df)
    agreement_rate = df["agreement"].mean() * 100

    logger.info(f"\n{'='*50}")
    logger.info("Key Metrics:")
    logger.info(f"  Exact agreement rate: {agreement_rate:.1f}%")
    logger.info(f"  Cohen's Kappa:        {kappa:.3f}")

    for method, summary in [("A", summary_a), ("B", summary_b)]:
        knows = (
            summary["overall"].get("strong-knows", 0)
            + summary["overall"].get("weak-knows", 0)
        )
        kr = knows / summary["total"] * 100 if summary["total"] > 0 else 0
        logger.info(f"  Method {method} knows rate: {kr:.1f}%")

    logger.info(f"\nAll outputs saved to: {output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()