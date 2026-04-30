"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.supervised import train as sft_train
from tinker_cookbook.recipes.chat_sl.chat_datasets import Tulu3Builder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio
from dataclasses import dataclass
from typing import Optional

@dataclass
class SFTConfig:
    """Hyperparameters and settings for a Tinker SFT run."""
    # model
    model_name: str = "meta-llama/Llama-3.2-1B"
 
    # data
    dataset: str = "tulu3"
    max_length: int = 4096
    batch_size: int = 128
 
    # optimizer
    learning_rate: float = 5e-4
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


class SFTTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args

    def build_dataset(self, dataset_name: str, model_name: str, renderer_name: str, max_length: int, batch_size: int):
        """
        Return a Tinker dataset builder for the given dataset name.
        """
        common_cfg = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        )
    
        if dataset_name == "tulu3":
            return Tulu3Builder(common_config=common_cfg)
        else:
            raise ValueError(
                f"unknown dataset '{dataset_name}'. "
            )
    
    def train(self):
        a = self.training_args
 
        # name the training run
        run_name = f"sft-r{a.lora_rank}-lr{a.learning_rate}-bs{a.batch_size}"
        log_path = a.log_path if a.log_path is not None else f"logs/{run_name}"
 
      
 
        # build dataset
        dataset_builder = self.build_dataset(
            dataset_name=a.dataset,
            model_name=a.model_name,
            renderer_name="role_colon",
            max_length=a.max_length,
            batch_size=a.batch_size,
        )
        
        # config
        train_config = sft_train.Config(
            log_path=a.log_path,
            model_name=a.model_name,
            renderer_name="role_colon",
            load_checkpoint_path=a.load_checkpoint_path,
            dataset_builder=dataset_builder,
            learning_rate=a.learning_rate,
            lr_schedule=a.lr_schedule,
            num_epochs=a.num_epochs,
            lora_rank=a.lora_rank,
            save_every=a.save_every,
            eval_every=a.eval_every,
            max_steps=a.max_steps,
        )
 
        asyncio.run(sft_train.main(train_config))