import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from train_sft import SFTTrainer


def format_example(example):
    prompt = example.get("prompt", "") or example.get("instruction", "") or example.get("input", "")
    response = example.get("response", "") or example.get("output", "")
    return {"text": f"User: {prompt}\nAssistant: {response}"}


def tokenize_example(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_name = "allenai/tulu-3-sft-olmo-2-mixture-0225"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = load_dataset(dataset_name)

    small_train = dataset["train"].select(range(500))
    train_dataset = small_train.map(format_example)
    train_dataset = train_dataset.map(lambda x: tokenize_example(x, tokenizer))

    columns_to_keep = ["input_ids", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns_to_keep)

    training_args = TrainingArguments(
        output_dir="./sft_checkpoint",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=None,
        training_args=training_args
    )

    trainer.train()


if __name__ == "__main__":
    main()