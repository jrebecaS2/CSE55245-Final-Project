"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""

from copy import deepcopy

from transformers import Trainer, default_data_collator


class SFTTrainer:
    REQUIRED_COLUMNS = {"input_ids", "attention_mask", "labels"}

    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args

    def _validate_dataset(self, dataset, split_name):
        if dataset is None:
            return

        column_names = set(getattr(dataset, "column_names", []))
        missing_columns = self.REQUIRED_COLUMNS - column_names
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{split_name} dataset is missing required columns: {missing}. "
                "Preprocess the dataset to include assistant-only labels before training."
            )

    def _build_training_args(self):
        args = deepcopy(self.training_args)

        if self.val_dataset is None:
            if hasattr(args, "eval_strategy"):
                args.eval_strategy = "no"
            if hasattr(args, "do_eval"):
                args.do_eval = False
            if hasattr(args, "load_best_model_at_end"):
                args.load_best_model_at_end = False
        elif hasattr(args, "do_eval"):
            args.do_eval = True

        return args

    def train(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self._validate_dataset(self.train_dataset, "train")
        self._validate_dataset(self.val_dataset, "validation")

        trainer_args = self._build_training_args()

        trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=default_data_collator,
        )

        train_result = trainer.train()
        trainer.save_model(trainer_args.output_dir)
        trainer.save_state()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        if self.val_dataset is not None:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        self.tokenizer.save_pretrained(trainer_args.output_dir)

        return trainer