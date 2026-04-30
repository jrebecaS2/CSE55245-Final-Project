import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_pref import PREFConfig, PREFTrainer
import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Config loaded: {path}")
    return cfg

def build_training_args(cfg: dict) -> PREFConfig:
    """Read hyperparameters from the config dict into an SFTConfig dataclass."""
    return PREFConfig(
         # model
        model_name=cfg.get("model_name", "meta-llama/Llama-3.2-1B"),
        filter_dataset = cfg.get("filter_dataset", False),
        
        load_checkpoint_path=cfg.get("load_checkpoint_path", None),
 
        # data
        dataset=cfg.get("dataset", "olmo2_preference"),
        max_length=cfg.get("max_length", 4096),
        batch_size=cfg.get("batch_size", 128),
 
        # dpo
        dpo_beta=cfg.get("dpo_beta", 0.1),
        learning_rate=cfg.get("learning_rate", 1e-5),
        lr_schedule=cfg.get("lr_schedule", "linear"),
        num_epochs=cfg.get("num_epochs", 1),
        max_steps=cfg.get("max_steps", None),
 
        # lora
        lora_rank=cfg.get("lora_rank", 16),
 
        # checkpoint
        save_every=cfg.get("save_every", 500),
        eval_every=cfg.get("eval_every", 500),
        log_path =cfg.get("log_path", None)
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT training via Tinker.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file",)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
 
    training_args = build_training_args(cfg)
    print(f"Resuming from checkpoint: {training_args.load_checkpoint_path}")
    
    print(training_args)
    trainer = PREFTrainer(
        model=None,
        tokenizer=None,
        train_dataset=None,
        val_dataset=None,
        training_args=training_args,
    )
    trainer.train()

if __name__ == "__main__":
    main()