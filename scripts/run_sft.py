import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from train_sft import SFTTrainer


MAX_LENGTH = 2048

ASSISTANT_HEADER = "<|assistant|>"


def tokenize_with_assistant_labels(example, tokenizer):
    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize the full conversation
    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    offsets = tokenized["offset_mapping"]

    assistant_char_ranges = []
    search_start = 0
    for msg in messages:
        if msg["role"] == "assistant":
            # Find the header that precedes this assistant turn
            header_pos = full_text.find(ASSISTANT_HEADER, search_start)
            if header_pos == -1:
                continue
            # The content starts after the header + newline
            content_start = header_pos + len(ASSISTANT_HEADER)
            # Skip the newline right after the header if present
            if content_start < len(full_text) and full_text[content_start] == "\n":
                content_start += 1
            # Find where this content ends (next header or end of string)
            content_end = full_text.find("<|", content_start)
            if content_end == -1:
                content_end = len(full_text)
            assistant_char_ranges.append((content_start, content_end))
            search_start = content_end
        else:
            header_tag = f"<|{msg['role']}|>"
            tag_pos = full_text.find(header_tag, search_start)
            if tag_pos != -1:
                search_start = tag_pos + len(header_tag)

    labels = [-100] * len(input_ids)
    for token_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue  
        for char_start, char_end in assistant_char_ranges:
            if tok_start >= char_start and tok_end <= char_end:
                labels[token_idx] = input_ids[token_idx]
                break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_name = "allenai/tulu-3-sft-olmo-2-mixture-0225"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="bfloat16")

    dataset = load_dataset(dataset_name)

    full_train = dataset["train"]

    train_dataset = full_train.map(
        lambda ex: tokenize_with_assistant_labels(ex, tokenizer),
        remove_columns=full_train.column_names,
        num_proc=4,
    )

    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns_to_keep)

    sample = train_dataset[0]
    decoded_all = tokenizer.decode(sample["input_ids"])
    label_ids = [tid for tid in sample["labels"].tolist() if tid != -100]
    decoded_labels = tokenizer.decode(label_ids) if label_ids else "<empty>"
    print("SANITY CHECK — first training example")
    print("FULL TEXT:\n", decoded_all[:500], "...")
    print("ASSISTANT-ONLY (what loss sees):\n", decoded_labels[:500], "...")

    training_args = TrainingArguments(
        output_dir="./sft_checkpoint",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,      # effective batch size = 2 * 16 = 32
        num_train_epochs=2,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,         # saves ~40% VRAM
        dataloader_num_workers=4,
        report_to="none",
        weight_decay=0.1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=None,
        training_args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()