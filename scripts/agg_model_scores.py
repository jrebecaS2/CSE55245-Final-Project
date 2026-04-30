import os
import json
from collections import defaultdict

# aggregate evaluation metrics across all models + tasks

ROOT = "scripts"
MODELS = ["dpo-full", "dpo-300", "dpo-filtered", "dpo-beta01", "dpo-beta5"]
TASKS = ["gsm8k-metrics", "ifeval-metrics", "mbpp-metrics"]

results = {m: {t: [] for t in TASKS} for m in MODELS}

def extract_model_name(filename):
    """
    Given a filename like 'gsm8k-dpo-beta01-100.json',
    return the model key: 'dpo-beta01'
    """
    for m in MODELS:
        if m in filename:
            return m
    return None

# Walk through each task directory
for task in TASKS:
    task_dir = os.path.join(ROOT, task)
    if not os.path.isdir(task_dir):
        continue

    for fname in os.listdir(task_dir):
        if not fname.endswith(".json"):
            continue

        model = extract_model_name(fname)
        if model is None:
            continue

        fpath = os.path.join(task_dir, fname)
        with open(fpath, "r") as f:
            obj = json.load(f)

        # Extract primary score
        try:
            score = obj["metrics"]["primary_score"]
            results[model][task].append(score)
        except KeyError:
            print(f"Warning: missing primary_score in {fpath}")

# Build final summary object
summary = {}

for model in MODELS:
    model_obj = {}
    task_scores = []

    for task in TASKS:
        scores = results[model][task]
        avg = sum(scores) / len(scores) if scores else None
        model_obj[task] = {
            "scores": scores,
            "average": avg
        }
        if avg is not None:
            task_scores.append(avg)

    # Overall average across tasks
    overall_avg = sum(task_scores) / len(task_scores) if task_scores else None
    model_obj["overall_average"] = overall_avg

    summary[model] = model_obj

# Print summary object
import pprint
pprint.pprint(summary)
