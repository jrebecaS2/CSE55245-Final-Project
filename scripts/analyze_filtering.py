"""
exploration 1: data filtering

  1. sumarizes what was filtered and why (from filtered_out_examples.jsonl)
  2. categorizes filtered examples by prompt type
  4. produces a qualitative report with 3-5 examples per category
  5. creates comparison plots
"""

import argparse
import json
import re
import os
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# torch/transformers only needed for inference — imported lazily below
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


# ── Palette ────────────────────────────────────────────────────────────────────
COLORS = {
    "unfiltered": "#378ADD",
    "filtered":   "#1D9E75",
    "neutral":    "#888780",
    "bg":         "#F4F6FA",
    "grid":       "#D3D1C7",
    "text":       "#2C2C2A",
}

# ── Prompt type classifier ─────────────────────────────────────────────────────
# Simple heuristic rules — no external model needed

CODING_PATTERNS = [
    r"\bcode\b", r"\bpython\b", r"\bjavascript\b", r"\bhtml\b", r"\bfunction\b",
    r"\bscript\b", r"\bprogram\b", r"\bclass\b", r"\bloop\b", r"\barray\b",
    r"\bsql\b", r"\bdebug\b",
]
MATH_PATTERNS = [
    r"\bmath\b", r"\bcalculate\b", r"\bsolve\b", r"\bequation\b", r"\bnumber\b",
    r"\bproof\b", r"\bderivative\b", r"\bintegral\b", r"\bgeometry\b",
    r"\d+\s*[\+\-\*\/]\s*\d+",
]
SAFETY_PATTERNS = [
    r"\bharm\b", r"\billegal\b", r"\bdrug\b", r"\bweapon\b", r"\bkill\b",
    r"\bdanger\b", r"\bsuicid\b", r"\battack\b", r"\bexploit\b", r"\bhack\b",
]
INSTRUCTION_PATTERNS = [
    r"\blist\b", r"\bsummariz\b", r"\bexplain\b", r"\bdescrib\b", r"\bwrit[e ]\b",
    r"\btranslat\b", r"\bgenerat\b", r"\bformat\b", r"\bstep[s ]?\b",
]


def classify_prompt(prompt: str) -> str:
    p = prompt.lower()
    if any(re.search(pat, p) for pat in SAFETY_PATTERNS):
        return "safety"
    if any(re.search(pat, p) for pat in CODING_PATTERNS):
        return "coding"
    if any(re.search(pat, p) for pat in MATH_PATTERNS):
        return "math"
    if any(re.search(pat, p) for pat in INSTRUCTION_PATTERNS):
        return "instruction-following"
    return "general"


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_reason(reason: str) -> str:
    if reason.startswith("near_identical"):
        return "near_identical"
    return reason


def token_len(text: str) -> int:
    return len(text.split())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def seq_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.split(), b.split()).ratio()


def wrap(text: str, width: int = 88) -> str:
    import textwrap
    return "\n    ".join(textwrap.wrap(text, width))


# ── Load filtered examples ─────────────────────────────────────────────────────

def load_filtered(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ex["reason_clean"] = normalize_reason(ex.get("reason", "unknown"))
            ex["prompt_type"] = classify_prompt(ex.get("prompt", ""))
            examples.append(ex)
    return examples


def sample_examples(
    examples: list[dict],
    n_per_cell: int = 3,
) -> list[dict]:
    """
    Sample n_per_cell examples per (reason, prompt_type) cell.
    Spreads samples evenly across the dataset rather than just taking first N.
    """
    by_cell = defaultdict(list)
    for ex in examples:
        key = (ex["reason_clean"], ex["prompt_type"])
        by_cell[key].append(ex)

    sampled = []
    for (reason, ptype), items in by_cell.items():
        if len(items) <= n_per_cell:
            sampled.extend(items)
        else:
            step = len(items) // n_per_cell
            sampled.extend(items[i * step] for i in range(n_per_cell))
    return sampled


# ── Inference ─────────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, path: str, tokenizer, max_new_tokens: int):
        print(f"  Loading {path} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16
        )
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            return decoded[len(prompt):].strip()
        return decoded.strip()


def is_refusal(text: str) -> bool:
    """Heuristic: does this response look like a refusal?"""
    refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable",
        "i won't", "i will not", "i'm not able", "i don't think i should",
        "i'm sorry, but", "sorry, i can't", "as an ai",
        "i must decline", "that's not something i",
    ]
    t = text.lower()
    return any(phrase in t for phrase in refusal_phrases)


def run_inference(
    sampled: list[dict],
    unfiltered: Runner,
    filtered: Runner,
) -> list[dict]:
    results = []
    total = len(sampled)
    for i, ex in enumerate(sampled, 1):
        prompt = ex["prompt"]
        print(f"  [{i}/{total}] {prompt[:60]}...")

        resp_unf = unfiltered.generate(prompt)
        resp_flt = filtered.generate(prompt)

        results.append({
            **ex,
            "resp_unfiltered":         resp_unf,
            "resp_filtered":           resp_flt,
            "sim_models":              round(seq_sim(resp_unf, resp_flt), 4),
            "unf_is_refusal":          is_refusal(resp_unf),
            "flt_is_refusal":          is_refusal(resp_flt),
            "unf_len":                 token_len(resp_unf),
            "flt_len":                 token_len(resp_flt),
            "chosen_len":              token_len(ex.get("chosen", "")),
            "rejected_len":            token_len(ex.get("rejected", "")),
            "chosen_rejected_jaccard": round(jaccard(ex.get("chosen",""), ex.get("rejected","")), 4),
        })
    return results


# ── Qualitative report ─────────────────────────────────────────────────────────

QUALITATIVE_LIMIT = 5   # max examples to print per (reason, prompt_type) cell

def verdict(r: dict) -> str:
    """Simple heuristic verdict for each example."""
    unf_ref = r["unf_is_refusal"]
    flt_ref = r["flt_is_refusal"]
    sim     = r["sim_models"]

    if unf_ref and not flt_ref:
        return "FILTERING HELPED — unfiltered refused, filtered answered"
    if not unf_ref and flt_ref:
        return "FILTERING HURT — filtered refused, unfiltered answered"
    if sim > 0.85:
        return "NO DIFFERENCE — responses nearly identical"
    if r["flt_len"] > r["unf_len"] * 1.25:
        return "FILTERING HELPED — filtered gave more detailed response"
    if r["flt_len"] < r["unf_len"] * 0.75:
        return "FILTERING HURT — filtered gave shorter response"
    return "SUBTLE DIFFERENCE — inspect manually"


def print_qualitative_report(results: list[dict], output_path: Path):
    by_cell = defaultdict(list)
    for r in results:
        by_cell[(r["reason_clean"], r["prompt_type"])].append(r)

    lines = []
    lines.append("=" * 100)
    lines.append("EXPLORATION 1: QUALITATIVE ANALYSIS — FILTERED vs UNFILTERED DPO")
    lines.append("=" * 100)
    lines.append("")
    lines.append("This report shows examples from the filtered-out training set and how")
    lines.append("removing them affected model responses on the same prompts.")
    lines.append("")

    shown = 0
    for (reason, ptype), items in sorted(by_cell.items()):
        lines.append(f"{'─'*100}")
        lines.append(f"  FILTER REASON: {reason.upper()}  |  PROMPT TYPE: {ptype.upper()}")
        lines.append(f"  {len(items)} examples in this cell — showing up to {QUALITATIVE_LIMIT}")
        lines.append(f"{'─'*100}")

        for r in items[:QUALITATIVE_LIMIT]:
            shown += 1
            lines.append(f"\n  Example {shown}")
            lines.append(f"  Filter reason : {r['reason']}")
            lines.append(f"  Prompt:\n    {wrap(r['prompt'][:300])}")
            lines.append(f"\n  Reference chosen  ({r['chosen_len']} tokens):")
            lines.append(f"    {wrap(r['chosen'][:250])}")
            lines.append(f"  Reference rejected ({r['rejected_len']} tokens):")
            lines.append(f"    {wrap(r['rejected'][:250])}")
            lines.append(f"  Chosen/rejected Jaccard similarity: {r['chosen_rejected_jaccard']:.3f}")

            lines.append(f"\n  Unfiltered DPO response ({r['unf_len']} tokens)  [refusal={r['unf_is_refusal']}]:")
            lines.append(f"    {wrap(r['resp_unfiltered'][:350])}")
            lines.append(f"\n  Filtered DPO response   ({r['flt_len']} tokens)  [refusal={r['flt_is_refusal']}]:")
            lines.append(f"    {wrap(r['resp_filtered'][:350])}")

            lines.append(f"\n  Model similarity : {r['sim_models']:.3f}")
            lines.append(f"  Verdict          : {verdict(r)}")
            lines.append("")

    report = "\n".join(lines)
    print(report)
    output_path.write_text(report)
    print(f"\nQualitative report saved to {output_path}")


# ── Example tables ────────────────────────────────────────────────────────────

def truncate(text: str, n: int) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + "\u2026" if len(text) > n else text


def save_example_tables(
    all_examples: list,
    output_dir,
    n: int = 5,
    results: list = None,
):
    """
    For each filter reason category, pick the top N most representative examples
    and write them as a plain-text table and a CSV.

    If inference results are provided, includes model response columns.

    Outputs:
        example_tables.txt   -- human-readable fixed-width tables
        example_tables.csv   -- importable into Excel / Google Sheets
    """
    result_by_prompt = {}
    if results:
        for r in results:
            result_by_prompt[r["prompt"]] = r

    by_reason = defaultdict(list)
    for ex in all_examples:
        by_reason[ex["reason_clean"]].append(ex)

    BASE_COLS = [
        ("Prompt type",  "prompt_type", 18),
        ("Prompt",       "prompt",      52),
        ("Chosen",       "chosen",      45),
        ("Rejected",     "rejected",    45),
        ("Filter reason","reason",      35),
    ]
    INF_COLS = [
        ("Unfiltered resp", "resp_unfiltered", 45),
        ("Filtered resp",   "resp_filtered",   45),
        ("Verdict",         "verdict",         38),
    ]
    has_inf = bool(results)
    cols = BASE_COLS + (INF_COLS if has_inf else [])

    txt_lines = []
    csv_rows  = [[ c[0] for c in cols ]]

    for reason in sorted(by_reason.keys()):
        items = by_reason[reason]
        if len(items) <= n:
            sampled = items
        else:
            step = len(items) // n
            sampled = [items[i * step] for i in range(n)]

        txt_lines.append("")
        txt_lines.append("=" * 120)
        txt_lines.append(
            f"  CATEGORY: {reason.upper()}  "
            f"({len(items)} total removed -- showing {len(sampled)})"
        )
        txt_lines.append("=" * 120)

        sep = ["u2500" * w for _, _, w in cols]
        sep = [chr(0x2500) * w for _, _, w in cols]
        txt_lines.append(chr(0x250C) + chr(0x252C).join(sep) + chr(0x2510))
        txt_lines.append(chr(0x2502) + chr(0x2502).join(
            label.ljust(w)[:w] for label, _, w in cols
        ) + chr(0x2502))
        row_sep = chr(0x251C) + chr(0x253C).join(sep) + chr(0x2524)
        txt_lines.append(row_sep)

        for ex in sampled:
            inf = result_by_prompt.get(ex["prompt"])
            cells = {}
            for _, key, width in BASE_COLS:
                cells[key] = truncate(str(ex.get(key, "")), width)
            if has_inf:
                if inf:
                    cells["resp_unfiltered"] = truncate(inf.get("resp_unfiltered","--"), 45)
                    cells["resp_filtered"]   = truncate(inf.get("resp_filtered","--"), 45)
                    cells["verdict"]         = truncate(verdict(inf), 38)
                else:
                    cells["resp_unfiltered"] = "(not sampled)"
                    cells["resp_filtered"]   = "(not sampled)"
                    cells["verdict"]         = "--"

            txt_lines.append(chr(0x2502) + chr(0x2502).join(
                cells.get(key,"").ljust(w)[:w] for _, key, w in cols
            ) + chr(0x2502))
            txt_lines.append(row_sep)

            csv_row = []
            for _, key, _ in BASE_COLS:
                csv_row.append(ex.get(key,"").replace("\n"," ").replace('"','""'))
            if has_inf:
                if inf:
                    csv_row.append(inf.get("resp_unfiltered","").replace("\n"," ").replace('"','""'))
                    csv_row.append(inf.get("resp_filtered","").replace("\n"," ").replace('"','""'))
                    csv_row.append(verdict(inf))
                else:
                    csv_row += ["(not sampled)","(not sampled)","--"]
            csv_rows.append(csv_row)

        txt_lines.append(chr(0x2514) + chr(0x2534).join(sep) + chr(0x2518))

    txt_path = output_dir / "example_tables.txt"
    
    #txt_path.write_text("\n".join(txt_lines))
    print("\n".join(txt_lines))
    print(f"\nExample tables saved to {txt_path}")

    csv_path = output_dir / "example_tables.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_rows:
            f.write(",".join(f'"{cell}"' for cell in row) + "\n")
    print(f"CSV saved to {csv_path}")



# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(all_examples: list[dict], results: list[dict]):
    total = len(all_examples)
    by_reason = Counter(ex["reason_clean"] for ex in all_examples)
    by_ptype  = Counter(ex["prompt_type"]  for ex in all_examples)

    print("\n" + "═" * 70)
    print("FILTERING SUMMARY")
    print("═" * 70)
    print(f"\nTotal examples removed by filtering: {total}")
    print("\nBy filter reason:")
    for reason, count in by_reason.most_common():
        print(f"  {reason:<30} {count:>6}  ({count/total*100:.1f}%)")
    print("\nBy prompt type (of removed examples):")
    for ptype, count in by_ptype.most_common():
        print(f"  {ptype:<30} {count:>6}  ({count/total*100:.1f}%)")

    if not results:
        return

    refusal_unf = sum(1 for r in results if r["unf_is_refusal"])
    refusal_flt = sum(1 for r in results if r["flt_is_refusal"])
    n = len(results)

    print(f"\n{'═'*70}")
    print("MODEL COMPARISON (on filtered-out prompts)")
    print(f"{'═'*70}")
    print(f"  Prompts analysed             : {n}")
    print(f"  Unfiltered model refusals    : {refusal_unf} ({refusal_unf/n*100:.1f}%)")
    print(f"  Filtered model refusals      : {refusal_flt} ({refusal_flt/n*100:.1f}%)")
    print(f"  Avg model similarity         : {np.mean([r['sim_models'] for r in results]):.3f}")
    print(f"  Avg unfiltered response len  : {np.mean([r['unf_len'] for r in results]):.1f} tokens")
    print(f"  Avg filtered response len    : {np.mean([r['flt_len'] for r in results]):.1f} tokens")

    verdicts = Counter(verdict(r) for r in results)
    print("\n  Verdict breakdown:")
    for v, c in verdicts.most_common():
        print(f"    {c:>3}x  {v}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_filter_breakdown(all_examples: list[dict], output_dir: Path):
    """Stacked bar: prompt types within each filter reason."""
    reasons = sorted(set(ex["reason_clean"] for ex in all_examples))
    ptypes  = sorted(set(ex["prompt_type"]  for ex in all_examples))

    data = {pt: [] for pt in ptypes}
    for reason in reasons:
        subset = [ex for ex in all_examples if ex["reason_clean"] == reason]
        counts = Counter(ex["prompt_type"] for ex in subset)
        for pt in ptypes:
            data[pt].append(counts.get(pt, 0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = np.zeros(len(reasons))
    palette = plt.cm.Set2(np.linspace(0, 1, len(ptypes)))
    for pt, color in zip(ptypes, palette):
        vals = np.array(data[pt])
        ax.bar(reasons, vals, bottom=bottoms, label=pt, color=color, width=0.55)
        bottoms += vals

    ax.set_xlabel("Filter reason", fontsize=11)
    ax.set_ylabel("Examples removed", fontsize=11)
    ax.set_title("Prompt types within each filter category", fontsize=12,
                 fontweight="bold", color=COLORS["text"], pad=10)
    ax.legend(title="Prompt type", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.9)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "filter_breakdown_by_prompt_type.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_refusal_rates(results: list[dict], output_dir: Path):
    """Grouped bar: refusal rate per prompt type, unfiltered vs filtered."""
    by_ptype = defaultdict(lambda: {"unf": [], "flt": []})
    for r in results:
        by_ptype[r["prompt_type"]]["unf"].append(int(r["unf_is_refusal"]))
        by_ptype[r["prompt_type"]]["flt"].append(int(r["flt_is_refusal"]))

    ptypes = sorted(by_ptype.keys())
    if not ptypes:
        return

    unf_rates = [np.mean(by_ptype[pt]["unf"]) * 100 for pt in ptypes]
    flt_rates = [np.mean(by_ptype[pt]["flt"]) * 100 for pt in ptypes]

    x = np.arange(len(ptypes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, unf_rates, w, label="Unfiltered DPO",
           color=COLORS["unfiltered"], alpha=0.85)
    ax.bar(x + w/2, flt_rates, w, label="Filtered DPO",
           color=COLORS["filtered"],   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ptypes, fontsize=10)
    ax.set_ylabel("Refusal rate (%)", fontsize=11)
    ax.set_title("Refusal rate by prompt type: unfiltered vs filtered", fontsize=12,
                 fontweight="bold", color=COLORS["text"], pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "refusal_rates_by_prompt_type.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_response_length(results: list[dict], output_dir: Path):
    """Grouped bar: avg response length per prompt type."""
    by_ptype = defaultdict(lambda: {"unf": [], "flt": []})
    for r in results:
        by_ptype[r["prompt_type"]]["unf"].append(r["unf_len"])
        by_ptype[r["prompt_type"]]["flt"].append(r["flt_len"])

    ptypes = sorted(by_ptype.keys())
    if not ptypes:
        return

    unf_lens = [np.mean(by_ptype[pt]["unf"]) for pt in ptypes]
    flt_lens = [np.mean(by_ptype[pt]["flt"]) for pt in ptypes]

    x = np.arange(len(ptypes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, unf_lens, w, label="Unfiltered DPO",
           color=COLORS["unfiltered"], alpha=0.85)
    ax.bar(x + w/2, flt_lens, w, label="Filtered DPO",
           color=COLORS["filtered"],   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(ptypes, fontsize=10)
    ax.set_ylabel("Avg response length (tokens)", fontsize=11)
    ax.set_title("Response length by prompt type: unfiltered vs filtered", fontsize=12,
                 fontweight="bold", color=COLORS["text"], pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "response_length_by_prompt_type.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_similarity_by_reason(results: list[dict], output_dir: Path):
    """Box plot: response similarity per filter reason."""
    by_reason = defaultdict(list)
    for r in results:
        by_reason[r["reason_clean"]].append(r["sim_models"])

    reasons = sorted(by_reason.keys())
    if not reasons:
        return
    data = [by_reason[r] for r in reasons]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, labels=reasons, patch_artist=True, widths=0.4,
                    medianprops={"color": COLORS["filtered"], "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["unfiltered"])
        patch.set_alpha(0.4)

    ax.axhline(0.85, color="#D85A30", linewidth=1.2, linestyle="--",
               label="0.85 near-identical threshold")
    ax.set_ylabel("Sequence similarity (unfiltered vs filtered)", fontsize=10)
    ax.set_title("How much did filtering change model responses?", fontsize=12,
                 fontweight="bold", color=COLORS["text"], pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "response_similarity_by_filter_reason.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_chosen_rejected_similarity(all_examples: list[dict], output_dir: Path):
    """
    Histogram of chosen/rejected Jaccard similarity for near_identical pairs.
    Shows what 'near identical' actually looked like in the data.
    """
    near = [ex for ex in all_examples if ex["reason_clean"] == "near_identical"]
    if not near:
        print("No near_identical examples — skipping similarity histogram.")
        return

    sims = [jaccard(ex.get("chosen",""), ex.get("rejected","")) for ex in near]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sims, bins=25, color=COLORS["unfiltered"], alpha=0.85, edgecolor="white")
    ax.axvline(0.9, color="#D85A30", linewidth=1.5, linestyle="--",
               label="Filter threshold (0.9)")
    ax.set_xlabel("Jaccard similarity (chosen vs rejected)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Similarity of near-identical pairs that were removed", fontsize=12,
                 fontweight="bold", color=COLORS["text"], pad=10)
    ax.legend(fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = output_dir / "near_identical_similarity_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_short_response_lengths(all_examples: list[dict], output_dir: Path):
    """
    Histogram of token lengths for the too-short responses that were removed.
    Separated into chosen_too_short and rejected_too_short.
    """
    chosen_short   = [ex for ex in all_examples if ex["reason_clean"] == "chosen_too_short"]
    rejected_short = [ex for ex in all_examples if ex["reason_clean"] == "rejected_too_short"]

    if not chosen_short and not rejected_short:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, items, label, color in [
        (axes[0], chosen_short,   "chosen_too_short",   COLORS["unfiltered"]),
        (axes[1], rejected_short, "rejected_too_short", COLORS["filtered"]),
    ]:
        if items:
            key = "chosen" if "chosen" in label else "rejected"
            lens = [token_len(ex.get(key, "")) for ex in items]
            ax.hist(lens, bins=20, color=color, alpha=0.85, edgecolor="white")
            ax.axvline(20, color="#D85A30", linewidth=1.5, linestyle="--",
                       label="Min token threshold (20)")
        ax.set_title(label, fontsize=10, fontweight="bold", color=COLORS["text"])
        ax.set_xlabel("Response length (tokens)", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_facecolor(COLORS["bg"])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Count", fontsize=10)
    fig.suptitle("Token length distribution of removed short responses",
                 fontsize=12, fontweight="bold", color=COLORS["text"])
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    p = output_dir / "short_response_length_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Exploration 1: filtering impact analysis.")
    p.add_argument("--filtered-log",         required=True,
                   help="Path to filtered_out_examples.jsonl")
    p.add_argument("--unfiltered",           default=None,
                   help="Path to unfiltered DPO merged model (optional — skip inference if omitted)")
    p.add_argument("--filtered-ckpt",        default=None,
                   help="Path to filtered DPO merged model (optional)")
    p.add_argument("--base-model",           default="meta-llama/Llama-3.2-1B")
    p.add_argument("--samples-per-cell",     type=int, default=3,
                   help="Examples to sample per (reason, prompt_type) cell")
    p.add_argument("--max-new-tokens",       type=int, default=120)
    p.add_argument("--output-dir",           default="analysis/filtering")
    p.add_argument("--data-only",            action="store_true",
                   help="Skip inference, only produce data analysis plots")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and inspect filtered examples ────────────────────────────────────
    print(f"\nLoading filtered examples from {args.filtered_log} ...")
    all_examples = load_filtered(args.filtered_log)
    total = len(all_examples)
    print(f"Loaded {total} filtered examples.")

    if total == 0:
        print("No examples found — check the path and file format.")
        return

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(all_examples, [])

    # ── Data-only plots (no model needed) ─────────────────────────────────────
    print("\nGenerating data analysis plots ...")
  #  plot_filter_breakdown(all_examples, output_dir)
  #  plot_chosen_rejected_similarity(all_examples, output_dir)
  #  plot_short_response_lengths(all_examples, output_dir)

    # ── Save filtered examples with prompt type labels ─────────────────────────
    labelled_path = output_dir / "filtered_examples_labelled.jsonl"
    with open(labelled_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Labelled examples saved to {labelled_path}")

    # ── Example tables (data-only, no inference) ──────────────────────────────
    print("\nGenerating example tables (data-only) ...")
    save_example_tables(all_examples, output_dir, n=5)

    # ── Inference (optional) ───────────────────────────────────────────────────
    run_inf = (
        not args.data_only
        and args.unfiltered is not None
        and args.filtered_ckpt is not None
    )

    results = []
    if run_inf:
        sampled = sample_examples(all_examples, args.samples_per_cell)
        print(f"\nSampled {len(sampled)} examples for inference "
              f"({args.samples_per_cell} per reason×prompt_type cell).")

        print("\nLoading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("\nLoading models ...")
        unf_runner = Runner(args.unfiltered,    tokenizer, args.max_new_tokens)
        flt_runner = Runner(args.filtered_ckpt, tokenizer, args.max_new_tokens)

        print("\nRunning inference ...")
        results = run_inference(sampled, unf_runner, flt_runner)

        # Save raw results
        results_path = output_dir / "inference_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Inference results saved to {results_path}")

        # Full summary with model comparison
        print_summary(all_examples, results)

        # Inference plots
        print("\nGenerating inference plots ...")
        plot_refusal_rates(results, output_dir)
        plot_response_length(results, output_dir)
        plot_similarity_by_reason(results, output_dir)

        # Qualitative report
        print("\nGenerating qualitative report ...")
        print_qualitative_report(results, output_dir / "qualitative_report.txt")

        # Example tables with inference columns
        print("\nGenerating example tables with model responses ...")
        save_example_tables(all_examples, output_dir, n=5, results=results)

    elif not run_inf and not args.data_only:
        print("\nSkipping inference — no model paths provided.")
        print("Re-run with --unfiltered and --filtered-ckpt to include model comparison.")
        print("Or use --data-only to suppress this message.")

    print(f"\nAll outputs saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()