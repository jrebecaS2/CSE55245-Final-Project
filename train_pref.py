"""
This module implements the PREFTrainer class for training your model using preference optimization.
"""
import asyncio
from dataclasses import dataclass
from typing import Optional, cast
from pathlib import Path
import datasets
import chz
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
)
from tinker_cookbook import renderers
import json
from collections import Counter

@dataclass
class PREFConfig:
    """Hyperparameters and settings for a Tinker DPO run."""
    # model
    model_name: str = "meta-llama/Llama-3.2-1B"
 
    # data
    dataset: str = "olmo2_preference"
    max_length: int = 4096
    batch_size: int = 128

    filter_dataset: Optional[bool] = False
 
    # dpo
    dpo_beta: float = 0.1          
    learning_rate: float = 1e-5
    lr_schedule: str = "linear"
    num_epochs: int = 1
    max_steps: Optional[int] = None
 
    # lora
    lora_rank: int = 16
 
    # checkpointing
    save_every: int = 500
    eval_every: int = 500
    load_checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None

@chz.chz
class Olmo2ComparisonBuilder(ComparisonDatasetBuilder):
    """Olmo2_Preference preference dataset comparison builder."""

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "allenai/olmo-2-0425-1b-preference-mix", split="train"
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)
        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        instruction = example["chosen"][0]["content"]
        chosen_response = example["chosen"][1]["content"]
        rejected_response = example["rejected"][1]["content"]

        prompt_conversation: list[renderers.Message] = [{"role": "user", "content": instruction}]

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        return LabeledComparison(comparison=comparison, label="A")
@chz.chz
class FilteredOlmo2ComparisonBuilder(ComparisonDatasetBuilder):
    """
    Filtered version of Olmo2ComparisonBuilder for Exploration 1.
    
    Removes preference pairs where:
      - Either response is extremely short (< min_tokens tokens)
      - chosen and rejected are nearly identical (Jaccard > similarity_threshold)
    
    Saves removed examples to logs/filtered_out_examples.jsonl for analysis.
    """

    min_tokens: int = 20
    similarity_threshold: float = 0.9
    filter_log_path: str = "logs/filtered_out_examples.jsonl"

    def _filter_reason(self, chosen: str, rejected: str) -> str | None:
        """Return the reason this pair should be filtered, or None if it should be kept."""
        chosen_tokens = chosen.split()
        rejected_tokens = rejected.split()

        if len(chosen_tokens) < self.min_tokens:
            return "chosen_too_short"
        if len(rejected_tokens) < self.min_tokens:
            return "rejected_too_short"

        chosen_set = set(chosen_tokens)
        rejected_set = set(rejected_tokens)
        if chosen_set and rejected_set:
            jaccard = len(chosen_set & rejected_set) / len(chosen_set | rejected_set)
            if jaccard > self.similarity_threshold:
                return f"near_identical (jaccard={jaccard:.2f})"

        return None

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        dataset = datasets.load_dataset(
            "allenai/olmo-2-0425-1b-preference-mix", split="train"
        )
        dataset = cast(datasets.Dataset, dataset)
        dataset = dataset.shuffle(seed=0)
        test_dataset = dataset.take(1024)
        train_dataset = dataset.skip(1024)

        # Apply filters and log removed examples
        kept, removed = [], []
        for example in train_dataset:
            chosen_response  = example["chosen"][1]["content"]
            rejected_response = example["rejected"][1]["content"]
            reason = self._filter_reason(chosen_response, rejected_response)

            if reason is not None:
                removed.append({
                    "prompt":   example["chosen"][0]["content"],
                    "chosen":   chosen_response,
                    "rejected": rejected_response,
                    "reason":   reason,
                })
            else:
                kept.append(example)

       

        # Save removed examples for qualitative analysis (Exploration 1)
        Path(self.filter_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filter_log_path, "w") as f:
            for ex in removed:
                f.write(json.dumps(ex) + "\n")
       
        # Convert kept list back to a HuggingFace dataset
        filtered_dataset = datasets.Dataset.from_list(kept)
        return filtered_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        # Identical to Olmo2ComparisonBuilder
        instruction      = example["chosen"][0]["content"]
        chosen_response  = example["chosen"][1]["content"]
        rejected_response = example["rejected"][1]["content"]

        prompt_conversation: list[renderers.Message] = [{"role": "user", "content": instruction}]

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=[{"role": "assistant", "content": chosen_response}],
            completion_B=[{"role": "assistant", "content": rejected_response}],
        )
        return LabeledComparison(comparison=comparison, label="A")
class PREFTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args

    def build_dataset( self,
        dataset_name: str,
        model_name: str,
        renderer_name: str,
        max_length: int,
        batch_size: int,
        cfg: PREFConfig,
    ):
        """
        Return a Tinker DPO dataset builder.
        """
        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        )
    
        if dataset_name == "olmo2_preference":
            if cfg.filter_dataset:
                comparison_builder = FilteredOlmo2ComparisonBuilder()
            else:
                comparison_builder = Olmo2ComparisonBuilder()
    
            return DPODatasetBuilderFromComparisons(
                common_config=common_config,
                comparison_builder=comparison_builder,
            )
        else:
            raise ValueError(f"unknown dataset '{dataset_name}'.")
    

    def train(self):
        a = self.training_args
 
        filter_tag = "-filtered" if a.filter_dataset else ""
        run_name = f"dpo-b{a.dpo_beta}-r{a.lora_rank}-lr{a.learning_rate}{filter_tag}"
        
        log_path = a.log_path if a.log_path is not None else f"logs/{run_name}"
        Path(log_path).mkdir(parents=True, exist_ok=True)
 
        # ── Resolve renderer ───────────────────────────────────────────────────
        renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
            model_name=a.model_name,
            explicit_renderer_name=None,
            load_checkpoint_path=a.load_checkpoint_path,
            base_url=None,
        )
 
 
        # ── Assemble Tinker DPO config ─────────────────────────────────────────
        config = train_dpo.Config(
            log_path=log_path,
            model_name=a.model_name,
            renderer_name=renderer_name,
            dataset_builder=self.build_dataset(
                dataset_name=a.dataset,
                model_name=a.model_name,
                renderer_name=renderer_name,
                max_length=a.max_length,
                batch_size=a.batch_size,
                cfg=a,
            ),
            load_checkpoint_path=a.load_checkpoint_path,
            learning_rate=a.learning_rate,
            lr_schedule=a.lr_schedule,
            num_epochs=a.num_epochs,
            dpo_beta=a.dpo_beta,         
            lora_rank=a.lora_rank,
            max_steps=a.max_steps,
        )
 
        train_dpo.main(config)