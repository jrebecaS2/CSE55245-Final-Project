"""
This module implements the SFTTrainer class for training your model using supervised fine-tuning (SFT).
"""

from datasets import load_dataset
from transformers import (
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)


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


def format_example(example):
    prompt = example.get("prompt", "") or example.get("instruction", "") or example.get("input", "")
    response = example.get("response", "") or example.get("output", "")
    return {"text": f"User: {prompt}\nAssistant: {response}"}


def tokenize_example(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


if __name__ == "__main__":
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_name = "allenai/tulu-3-sft-olmo-2-mixture-0225"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"].map(format_example)

    if "validation" in dataset:
        val_dataset = dataset["validation"].map(format_example)
    else:
        val_dataset = None

    train_dataset = train_dataset.map(lambda x: tokenize_example(x, tokenizer))
    if val_dataset is not None:
        val_dataset = val_dataset.map(lambda x: tokenize_example(x, tokenizer))

    columns_to_keep = ["input_ids", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns_to_keep)
    if val_dataset is not None:
        val_dataset.set_format(type="torch", columns=columns_to_keep)

    training_args = TrainingArguments(
        output_dir="./sft_checkpoint",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args
    )

    trainer.train()