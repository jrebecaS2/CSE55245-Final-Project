"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""

from transformers import Trainer, DataCollatorForLanguageModeling


class SFTTrainer:
    def __init__(self, model, tokenizer, train_dataset, val_dataset, training_args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args

    def train(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(self.training_args.output_dir)
        self.tokenizer.save_pretrained(self.training_args.output_dir)

        return trainer