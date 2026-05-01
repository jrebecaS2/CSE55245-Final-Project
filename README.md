# CSE 5525 Final Project

## Overview

This repository contains code for the CSE 5525 Default Project.

The project includes:
- supervised fine-tuning (SFT)
- preference optimization / DPO training
- evaluation wrappers for GSM8K, IFEval, and MBPP
- exploration for data filtering and alignment strength explorations

## Repository Layout

- train_sft.py — SFT trainer implementation using Tinker
- train_pref.py — DPO / preference optimization trainer and filtered dataset builder
- merge_model.py — helper for merging a PEFT adapter into a full model checkpoint
- configs/ — YAML training configuration files
- scripts/ — run wrappers and analysis utilities
- explorations/ — metric result directories for analysis
- evals/ — OLMES evaluation harness

## Running Training

### Supervised Fine-Tuning (SFT)

Run SFT with a YAML config:

`powershell
python scripts/run_sft.py --config <config-file>
`

To resume training from a checkpoint:

`powershell
python scripts/run_sft.py --config <config-file> --checkpoint <checkpoint-dir>
`

### Preference Optimization (DPO)

Run DPO training with a YAML config:

`powershell
python scripts/run_pref.py --config <config-file>
`

### Merge Adapters

After training a PEFT adapter, merge it into a full checkpoint using the generic `merge_model.py` helper.

`powershell
python merge_model.py --base-model <base-model> --adapter <adapter-path> --output-dir <output-dir>
`

Optional download from Tinker

`powershell
python merge_model.py --base-model <base-model> --adapter <adapter-path> --output-dir <output-dir> --tinker-download-path <tinker-uri>
`

## Config Files

Current configs:
- configs/sft-1.yaml (SFT baseline)
- configs/dpo-1.yaml (SFT+DPO)
- configs/dpo-b-low.yaml (exploration 2)
- configs/dpo-b-high.yaml (exploration 2)
- configs/dpo-filtered.yaml (exploration 1)

These files define model settings, optimizer parameters, DPO beta values, and logging/checkpoint behavior.

## Evaluation

The repo includes an evaluation harness under evals/.

## Exploration Scripts

These scripts support analysis of filtering, model comparisons, and score aggregation.

### scripts/analyze_filtering.py

Analyzes preference-data filtering behavior and produces:
- filtered example summaries and reasons
- prompt-type classification (safety, coding, math, instruction-following, general)
- qualitative examples for each filter category
- comparison plots for filtered vs. unfiltered behavior
- optional inference using saved model checkpoints

### scripts/compare_filter_results.py

Compares two IFEval prediction files and produces:
- score comparisons at prompt and instruction levels
- instruction-type breakdowns
- examples where filtering changed performance
- qualitative response-difference analysis

### scripts/agg_model_scores.py

Aggregates evaluation metrics from multiple exploration directories.
- reads JSON results in explorations/gsm8k-metrics/, explorations/ifeval-metrics/, explorations/mbpp-metrics/
- collects primary_score values per model
- computes task averages and overall model averages

### scripts/plots.py

Provides plotting helpers for:
- DPO beta sweep visualizations
- filtered vs unfiltered model comparisons

### Training wrappers

- scripts/run_sft.py — loads a YAML config and runs SFT training
- scripts/run_pref.py — loads a YAML config and runs preference/DPO training

## Workflow

1. Run baseline SFT: python scripts/run_sft.py --config configs/sft-1.yaml
2. Run preference/DPO training: python scripts/run_pref.py --config configs/dpo-1.yaml
3. Run DPO training with explorations using configs dpo-b-low. dpo-b-high, dpo-filtered
4. Merge the adapter 
5. Run evaluations through evals/run_eval.sh
6. Use analysis scripts in scripts/ to inspect filtering and compare results
