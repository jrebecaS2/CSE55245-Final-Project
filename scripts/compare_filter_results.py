"""
data-filtering instruction following analysis
compares two IFEval predictions.jsonl files and produces:
  1. score comparisons (prompt and instruction level acc)
  2. instruction-type breakdown (where filtering helped/hurt)
  3. examples where filtering changed the outcome
  4. qualitative analysis of response differences
"""

import argparse
import json
import re
import textwrap
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    "unfiltered": "#378ADD",
    "filtered":   "#1D9E75",
    "neutral":    "#888780",
    "worse":      "#D85A30",
    "bg":         "#F4F6FA",
    "grid":       "#D3D1C7",
    "text":       "#2C2C2A",
}

# maps metric key prefix → human readable label
INST_TYPE_LABELS = {
    "length_constraints":   "Length constraints",
    "keywords":             "Keyword requirements",
    "detectable_format":    "Format detection",
    "detectable_content":   "Content detection",
    "change_case":          "Case changes",
    "combination":          "Combination tasks",
    "punctuation":          "Punctuation rules",
    "startend":             "Start/end constraints",
    "language":             "Language constraints",
}


def inst_type(metric_key: str) -> str:
    """Extract instruction type from a metric key like 'keywords:existence_strict_acc'."""
    prefix = metric_key.split(":")[0]
    return INST_TYPE_LABELS.get(prefix, prefix)


def is_inst_metric(key: str) -> bool:
    """True if this is a per-instruction metric (not the aggregate prompt/inst level)."""
    return ":" in key and key not in (
        "prompt_level_strict_acc", "inst_level_strict_acc",
        "prompt_level_loose_acc",  "inst_level_loose_acc",
    )


# load

def load(path: str) -> dict[int, dict]:
    """Load predictions jsonl, keyed by doc_id."""
    data = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            data[row["doc_id"]] = row
    return data


# helpers

def get_response(row: dict) -> str:
    mo = row.get("model_output", [{}])
    if mo:
        return mo[0].get("model_answer", "").strip()
    return ""


def response_length(row: dict) -> int:
    return len(get_response(row).split())


def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.split(), b.split()).ratio()


def is_repetitive(text: str, threshold: float = 0.6) -> bool:
    """Heuristic: does the response repeat the same phrase over and over?"""
    words = text.split()
    if len(words) < 20:
        return False
    # Check if any 6-gram appears more than 3 times
    ngrams = [" ".join(words[i:i+6]) for i in range(len(words)-5)]
    counts = Counter(ngrams)
    most_common_count = counts.most_common(1)[0][1] if counts else 0
    return most_common_count > 3


def truncate(text: str, n: int = 300) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    return ("\n" + indent).join(textwrap.wrap(text.replace("\n", " "), width))


# aggregate scores

AGG_METRICS = [
    ("prompt_level_strict_acc", "Prompt strict"),
    ("prompt_level_loose_acc",  "Prompt loose"),
    ("inst_level_strict_acc",   "Instruction strict"),
    ("inst_level_loose_acc",    "Instruction loose"),
]


def aggregate_scores(data: dict) -> dict[str, float]:
    scores = defaultdict(list)
    for row in data.values():
        for key, _ in AGG_METRICS:
            if key in row["metrics"]:
                scores[key].append(row["metrics"][key])
    return {k: float(np.mean(v)) for k, v in scores.items()}


# instruction type breakdown

def inst_type_scores(data: dict) -> dict[str, dict]:
    """
    Returns dict mapping instruction_type ->
        {"unfiltered": mean_score, "n": count}
    over strict_acc metrics only.
    """
    by_type = defaultdict(list)
    for row in data.values():
        for key, val in row["metrics"].items():
            if is_inst_metric(key) and key.endswith("_strict_acc"):
                by_type[inst_type(key)].append(val)
    return {
        t: {"mean": float(np.mean(vals)), "n": len(vals)}
        for t, vals in by_type.items()
    }


# outcome classification per doc

def classify_change(unf_row: dict, flt_row: dict) -> str:
    """
    Classify how filtering changed the outcome for one prompt.
    Uses prompt_level_strict_acc as the primary signal.
    """
    u = unf_row["metrics"].get("prompt_level_strict_acc", None)
    f = flt_row["metrics"].get("prompt_level_strict_acc", None)
    if u is None or f is None:
        return "unknown"
    if u == 0 and f == 1:
        return "filtering_helped"
    if u == 1 and f == 0:
        return "filtering_hurt"
    if u == f:
        # Same outcome — check if responses differ meaningfully
        resp_u = get_response(unf_row)
        resp_f = get_response(flt_row)
        if seq_sim(resp_u, resp_f) < 0.6:
            return "different_response_same_score"
        return "no_change"
    return "no_change"


# qualitative analysis

def analyse_response_quality(row: dict) -> dict:
    resp = get_response(row)
    return {
        "length":      len(resp.split()),
        "repetitive":  is_repetitive(resp),
        "empty":       len(resp.strip()) == 0,
        "num_tokens":  row.get("model_output", [{}])[0].get("num_tokens", 0),
    }


def describe_instructions(metrics: dict) -> list[str]:
    """List which instructions the model passed/failed."""
    passed, failed = [], []
    for key, val in metrics.items():
        if is_inst_metric(key) and key.endswith("_strict_acc"):
            label = key.replace("_strict_acc", "").replace("_", " ")
            if val == 1.0:
                passed.append(label)
            else:
                failed.append(label)
    return passed, failed


def print_and_save_report(
    unf: dict, flt: dict,
    unf_scores: dict, flt_scores: dict,
    unf_inst: dict, flt_inst: dict,
    paired: list[dict],
    output_path: Path,
):
    lines = []

    def h(text): lines.append("\n" + "═" * 100 + f"\n  {text}\n" + "═" * 100)
    def sub(text): lines.append("\n" + "─" * 80 + f"\n  {text}\n" + "─" * 80)

    h("IFEVAL PREDICTIONS COMPARISON: UNFILTERED DPO vs FILTERED DPO")

    # ── 1. Aggregate scores ────────────────────────────────────────────────────
    sub("1. AGGREGATE SCORES")
    lines.append(f"\n  {'Metric':<35} {'Unfiltered':>12} {'Filtered':>12} {'Delta':>10}")
    lines.append(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*10}")
    for key, label in AGG_METRICS:
        u = unf_scores.get(key, 0.0)
        f = flt_scores.get(key, 0.0)
        d = f - u
        arrow = "↑" if d > 0.005 else ("↓" if d < -0.005 else "≈")
        lines.append(f"  {label:<35} {u:>12.3f} {f:>12.3f} {d:>+9.3f} {arrow}")

    # ── 2. Per-instruction-type breakdown ──────────────────────────────────────
    sub("2. PER-INSTRUCTION-TYPE BREAKDOWN (strict accuracy)")
    all_types = sorted(set(list(unf_inst.keys()) + list(flt_inst.keys())))
    lines.append(f"\n  {'Instruction type':<35} {'Unfiltered':>12} {'Filtered':>12} {'Delta':>10} {'N':>6}")
    lines.append(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*10} {'─'*6}")
    for t in all_types:
        u = unf_inst.get(t, {}).get("mean", float("nan"))
        f = flt_inst.get(t, {}).get("mean", float("nan"))
        n = unf_inst.get(t, {}).get("n", 0)
        d = f - u if not (np.isnan(u) or np.isnan(f)) else float("nan")
        arrow = "" if np.isnan(d) else ("↑" if d > 0.005 else ("↓" if d < -0.005 else "≈"))
        lines.append(
            f"  {t:<35} {u:>12.3f} {f:>12.3f} "
            f"{(f'{d:+.3f}' if not np.isnan(d) else '  n/a'):>10} "
            f"{arrow:>2} {n:>5}"
        )

    # ── 3. Outcome classification ──────────────────────────────────────────────
    sub("3. OUTCOME CHANGES")
    outcome_counts = Counter(r["change"] for r in paired)
    lines.append(f"\n  Total prompts compared: {len(paired)}")
    for outcome, count in outcome_counts.most_common():
        lines.append(f"  {outcome:<40} {count:>4}  ({count/len(paired)*100:.1f}%)")

    # ── 4. Response quality ────────────────────────────────────────────────────
    sub("4. RESPONSE QUALITY OVERVIEW")
    unf_rep = sum(1 for r in paired if r["unf_quality"]["repetitive"])
    flt_rep = sum(1 for r in paired if r["flt_quality"]["repetitive"])
    unf_len = np.mean([r["unf_quality"]["length"] for r in paired])
    flt_len = np.mean([r["flt_quality"]["length"] for r in paired])
    lines.append(f"\n  Repetitive responses:  unfiltered={unf_rep}/{len(paired)}  filtered={flt_rep}/{len(paired)}")
    lines.append(f"  Avg response length:   unfiltered={unf_len:.0f} tokens  filtered={flt_len:.0f} tokens")

    # ── 5. Qualitative examples ────────────────────────────────────────────────
    sub("5. QUALITATIVE EXAMPLES — WHERE FILTERING CHANGED BEHAVIOR")

    categories = [
        ("filtering_helped",              "FILTERING HELPED (unfiltered failed, filtered passed)"),
        ("filtering_hurt",                "FILTERING HURT (unfiltered passed, filtered failed)"),
        ("different_response_same_score", "DIFFERENT RESPONSE, SAME SCORE"),
    ]

    for outcome_key, outcome_label in categories:
        examples = [r for r in paired if r["change"] == outcome_key]
        if not examples:
            lines.append(f"\n  {outcome_label}: no examples found.")
            continue

        lines.append(f"\n  {outcome_label} ({len(examples)} total, showing up to 5)")
        shown = examples[:5]
        for i, ex in enumerate(shown, 1):
            doc_id = ex["doc_id"]
            unf_row = unf[doc_id]
            flt_row = flt[doc_id]

            passed_u, failed_u = describe_instructions(unf_row["metrics"])
            passed_f, failed_f = describe_instructions(flt_row["metrics"])

            lines.append(f"\n  Example {i}  (doc_id={doc_id})")
            lines.append(f"  Instructions passed  — unfiltered: {passed_u or ['none']}")
            lines.append(f"                         filtered:   {passed_f or ['none']}")
            lines.append(f"  Instructions failed  — unfiltered: {failed_u or ['none']}")
            lines.append(f"                         filtered:   {failed_f or ['none']}")
            lines.append(f"  Response similarity: {ex['sim']:.3f}")
            lines.append(f"  Unfiltered response ({ex['unf_quality']['length']} tokens)"
                         f"  [repetitive={ex['unf_quality']['repetitive']}]:")
            lines.append(f"    {wrap(truncate(get_response(unf_row), 400))}")
            lines.append(f"  Filtered response ({ex['flt_quality']['length']} tokens)"
                         f"  [repetitive={ex['flt_quality']['repetitive']}]:")
            lines.append(f"    {wrap(truncate(get_response(flt_row), 400))}")

    # ── 6. Notable patterns ────────────────────────────────────────────────────
    sub("6. NOTABLE PATTERNS")
    lines.append(generate_narrative(unf_scores, flt_scores, unf_inst, flt_inst,
                                    paired, outcome_counts))

    report = "\n".join(lines)
    print(report)
    output_path.write_text(report)
    print(f"\nReport saved to {output_path}")
    return report


def generate_narrative(unf_scores, flt_scores, unf_inst, flt_inst, paired, outcome_counts):
    """Auto-generate a short written summary of the findings."""
    lines = []

    prompt_strict_delta = flt_scores.get("prompt_level_strict_acc", 0) - \
                          unf_scores.get("prompt_level_strict_acc", 0)
    inst_strict_delta   = flt_scores.get("inst_level_strict_acc", 0) - \
                          unf_scores.get("inst_level_strict_acc", 0)

    if prompt_strict_delta > 0.01:
        lines.append(f"  Filtering improved prompt-level strict accuracy by "
                     f"{prompt_strict_delta:+.3f}, suggesting that removing noisy "
                     f"preference pairs helped the model better follow explicit instructions.")
    elif prompt_strict_delta < -0.01:
        lines.append(f"  Filtering reduced prompt-level strict accuracy by "
                     f"{abs(prompt_strict_delta):.3f}. This may indicate that some "
                     f"removed examples were providing useful training signal.")
    else:
        lines.append(f"  Prompt-level strict accuracy was largely unchanged after filtering "
                     f"(delta={prompt_strict_delta:+.3f}), suggesting filtering had limited "
                     f"impact on this metric.")

    # Instruction type deltas
    improved, degraded = [], []
    for t in set(list(unf_inst.keys()) + list(flt_inst.keys())):
        u = unf_inst.get(t, {}).get("mean", None)
        f = flt_inst.get(t, {}).get("mean", None)
        if u is not None and f is not None:
            d = f - u
            if d > 0.05:
                improved.append((t, d))
            elif d < -0.05:
                degraded.append((t, d))

    if improved:
        top = sorted(improved, key=lambda x: -x[1])[:3]
        lines.append(f"\n  Instruction types where filtering most helped: "
                     f"{', '.join(f'{t} ({d:+.2f})' for t,d in top)}")
    if degraded:
        top = sorted(degraded, key=lambda x: x[1])[:3]
        lines.append(f"\n  Instruction types where filtering most hurt: "
                     f"{', '.join(f'{t} ({d:+.2f})' for t,d in top)}")

    # Repetition
    unf_rep = sum(1 for r in paired if r["unf_quality"]["repetitive"])
    flt_rep = sum(1 for r in paired if r["flt_quality"]["repetitive"])
    if flt_rep < unf_rep:
        lines.append(f"\n  Filtered model produced fewer repetitive responses "
                     f"({flt_rep} vs {unf_rep}), consistent with removal of near-identical "
                     f"preference pairs that may have reinforced repetitive output patterns.")
    elif flt_rep > unf_rep:
        lines.append(f"\n  Filtered model produced more repetitive responses "
                     f"({flt_rep} vs {unf_rep}), which is unexpected and warrants "
                     f"further investigation.")

    helped = outcome_counts.get("filtering_helped", 0)
    hurt   = outcome_counts.get("filtering_hurt",   0)
    if helped > hurt:
        lines.append(f"\n  On balance, filtering helped more individual prompts than it hurt "
                     f"({helped} helped vs {hurt} hurt), though aggregate score changes "
                     f"may still be small.")
    elif hurt > helped:
        lines.append(f"\n  Filtering hurt more individual prompts than it helped "
                     f"({hurt} hurt vs {helped} helped), suggesting the removed examples "
                     f"contained useful training signal despite appearing noisy.")
    else:
        lines.append(f"\n  Filtering helped and hurt an equal number of individual prompts "
                     f"({helped} each), indicating mixed effects from the data cleaning.")

    return "\n".join(lines)


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_aggregate_scores(unf_scores, flt_scores, output_dir: Path):
    labels = [label for _, label in AGG_METRICS]
    unf_vals = [unf_scores.get(k, 0) for k, _ in AGG_METRICS]
    flt_vals = [flt_scores.get(k, 0) for k, _ in AGG_METRICS]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, unf_vals, w, label="Unfiltered DPO", color=C["unfiltered"], alpha=0.85)
    ax.bar(x + w/2, flt_vals, w, label="Filtered DPO",   color=C["filtered"],   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("IFEval aggregate scores: unfiltered vs filtered DPO",
                 fontsize=12, fontweight="bold", color=C["text"], pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=C["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "aggregate_scores.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_inst_type_breakdown(unf_inst, flt_inst, output_dir: Path):
    all_types = sorted(set(list(unf_inst.keys()) + list(flt_inst.keys())))
    if not all_types:
        return
    unf_vals = [unf_inst.get(t, {}).get("mean", 0) for t in all_types]
    flt_vals = [flt_inst.get(t, {}).get("mean", 0) for t in all_types]

    x = np.arange(len(all_types))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, unf_vals, w, label="Unfiltered DPO", color=C["unfiltered"], alpha=0.85)
    ax.bar(x + w/2, flt_vals, w, label="Filtered DPO",   color=C["filtered"],   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean strict accuracy", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-instruction-type accuracy: unfiltered vs filtered DPO",
                 fontsize=12, fontweight="bold", color=C["text"], pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=C["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "inst_type_breakdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_delta_by_inst_type(unf_inst, flt_inst, output_dir: Path):
    """Horizontal bar chart of deltas — makes improvements vs regressions obvious."""
    all_types = sorted(set(list(unf_inst.keys()) + list(flt_inst.keys())))
    deltas, colors = [], []
    for t in all_types:
        u = unf_inst.get(t, {}).get("mean", None)
        f = flt_inst.get(t, {}).get("mean", None)
        if u is not None and f is not None:
            d = f - u
            deltas.append(d)
            colors.append(C["filtered"] if d >= 0 else C["worse"])
        else:
            deltas.append(0)
            colors.append(C["neutral"])

    fig, ax = plt.subplots(figsize=(8, max(4, len(all_types) * 0.45)))
    bars = ax.barh(all_types, deltas, color=colors, height=0.55)
    ax.axvline(0, color=C["text"], linewidth=0.8)
    ax.set_xlabel("Delta (filtered − unfiltered)", fontsize=10)
    ax.set_title("Filtering effect per instruction type",
                 fontsize=12, fontweight="bold", color=C["text"], pad=10)
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="x", color=C["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    helped_patch = plt.Rectangle((0,0),1,1, color=C["filtered"], label="Filtering helped")
    hurt_patch   = plt.Rectangle((0,0),1,1, color=C["worse"],    label="Filtering hurt")
    ax.legend(handles=[helped_patch, hurt_patch], fontsize=9)
    plt.tight_layout()
    p = output_dir / "delta_by_inst_type.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_outcome_distribution(outcome_counts, output_dir: Path):
    labels = list(outcome_counts.keys())
    counts = [outcome_counts[l] for l in labels]
    color_map = {
        "filtering_helped":              C["filtered"],
        "filtering_hurt":                C["worse"],
        "different_response_same_score": C["unfiltered"],
        "no_change":                     C["neutral"],
    }
    colors = [color_map.get(l, C["neutral"]) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts, color=colors, width=0.5)
    ax.set_ylabel("Number of prompts", fontsize=10)
    ax.set_title("Per-prompt outcome of filtering", fontsize=12,
                 fontweight="bold", color=C["text"], pad=10)
    ax.set_xticklabels(
        [l.replace("_", "\n") for l in labels], fontsize=8
    )
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=C["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "outcome_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_response_length_comparison(paired, output_dir: Path):
    unf_lens = [r["unf_quality"]["length"] for r in paired]
    flt_lens = [r["flt_quality"]["length"] for r in paired]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(max(unf_lens), max(flt_lens)) + 10, 30)
    ax.hist(unf_lens, bins=bins, alpha=0.6, color=C["unfiltered"],
            label="Unfiltered DPO", edgecolor="white")
    ax.hist(flt_lens, bins=bins, alpha=0.6, color=C["filtered"],
            label="Filtered DPO",   edgecolor="white")
    ax.set_xlabel("Response length (tokens)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Response length distribution", fontsize=12,
                 fontweight="bold", color=C["text"], pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=C["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "response_length.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


# ── CSV export ─────────────────────────────────────────────────────────────────

def save_comparison_csv(unf, flt, paired, output_dir: Path):
    """Save a per-prompt comparison CSV importable into Excel/Sheets."""
    path = output_dir / "per_prompt_comparison.csv"
    rows = []
    header = [
        "doc_id", "change", "similarity",
        "unf_prompt_strict", "flt_prompt_strict",
        "unf_inst_strict",   "flt_inst_strict",
        "unf_len", "flt_len",
        "unf_repetitive", "flt_repetitive",
        "unf_response_preview", "flt_response_preview",
    ]
    rows.append(header)
    for r in paired:
        doc_id = r["doc_id"]
        unf_row = unf[doc_id]
        flt_row = flt[doc_id]
        rows.append([
            doc_id,
            r["change"],
            f"{r['sim']:.3f}",
            unf_row["metrics"].get("prompt_level_strict_acc", ""),
            flt_row["metrics"].get("prompt_level_strict_acc", ""),
            unf_row["metrics"].get("inst_level_strict_acc",   ""),
            flt_row["metrics"].get("inst_level_strict_acc",   ""),
            r["unf_quality"]["length"],
            r["flt_quality"]["length"],
            r["unf_quality"]["repetitive"],
            r["flt_quality"]["repetitive"],
            truncate(get_response(unf_row), 200).replace('"', '""'),
            truncate(get_response(flt_row), 200).replace('"', '""'),
        ])
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(f'"{c}"' for c in row) + "\n")
    print(f"CSV saved to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unfiltered", required=True,
                   help="Predictions jsonl from unfiltered DPO model")
    p.add_argument("--filtered",   required=True,
                   help="Predictions jsonl from filtered DPO model")
    p.add_argument("--output-dir", default="analysis/ifeval_comparison")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading unfiltered predictions: {args.unfiltered}")
    unf = load(args.unfiltered)
    print(f"Loading filtered predictions:   {args.filtered}")
    flt = load(args.filtered)

    common_ids = sorted(set(unf.keys()) & set(flt.keys()))
    print(f"Matched {len(common_ids)} prompts across both files.")

    if len(common_ids) == 0:
        print("No matching doc_ids found — check that both files cover the same eval set.")
        return

    unf_scores = aggregate_scores(unf)
    flt_scores = aggregate_scores(flt)
    unf_inst   = inst_type_scores(unf)
    flt_inst   = inst_type_scores(flt)

    paired = []
    for doc_id in common_ids:
        unf_row = unf[doc_id]
        flt_row = flt[doc_id]
        resp_u = get_response(unf_row)
        resp_f = get_response(flt_row)
        paired.append({
            "doc_id":      doc_id,
            "change":      classify_change(unf_row, flt_row),
            "sim":         seq_sim(resp_u, resp_f),
            "unf_quality": analyse_response_quality(unf_row),
            "flt_quality": analyse_response_quality(flt_row),
        })

    outcome_counts = Counter(r["change"] for r in paired)

    print_and_save_report(
        unf, flt,
        unf_scores, flt_scores,
        unf_inst, flt_inst,
        paired, output_dir / "comparison_report.txt",
    )

    print("\nGenerating plots ...")
    plot_aggregate_scores(unf_scores, flt_scores, output_dir)
    plot_inst_type_breakdown(unf_inst, flt_inst, output_dir)
    plot_delta_by_inst_type(unf_inst, flt_inst, output_dir)
    plot_outcome_distribution(outcome_counts, output_dir)
    plot_response_length_comparison(paired, output_dir)

    save_comparison_csv(unf, flt, paired, output_dir)

    print(f"\nAll outputs saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()