import sys
import os
import argparse

from evals.olmes.oe_eval.dependencies.BFCL.bfcl.model_handler import parser

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true",
                        help="Short 100-step run to verify training is working")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

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

    # ── Diagnostic: check label coverage across multiple examples ──
    print("=" * 60)
    print("LABEL DIAGNOSTIC — checking first 50 examples")
    print("=" * 60)
    zero_label = 0
    total_label_tokens = 0
    total_content_tokens = 0
    for i in range(min(50, len(train_dataset))):
        sample = train_dataset[i]
        labs = sample["labels"].tolist()
        n_labeled = sum(1 for l in labs if l != -100)
        n_content = sum(1 for m in sample["attention_mask"].tolist() if m == 1)
        total_label_tokens += n_labeled
        total_content_tokens += n_content
        if n_labeled == 0:
            zero_label += 1
        if i < 5:
            decoded_all = tokenizer.decode(sample["input_ids"])
            label_ids = [tid for tid in labs if tid != -100]
            decoded_labels = tokenizer.decode(label_ids) if label_ids else "<empty>"
            print(f"\nExample {i}: {n_labeled}/{n_content} tokens labeled")
            print(f"  FULL TEXT (first 300 chars): {decoded_all[:300]}")
            print(f"  LABELED TEXT (first 300 chars): {decoded_labels[:300]}")

    avg_labeled = total_label_tokens / min(50, len(train_dataset))
    avg_content = total_content_tokens / min(50, len(train_dataset))
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {zero_label}/50 examples have ZERO labels")
    print(f"  Avg labeled tokens per example: {avg_labeled:.1f}")
    print(f"  Avg content tokens per example: {avg_content:.1f}")
    print(f"  Label ratio: {avg_labeled/max(avg_content,1)*100:.1f}%")
    print(f"{'=' * 60}")
    if avg_labeled < 10:
        print("WARNING: Very few labels per example! The labeling logic is likely broken.")
        print("Training loss will appear near-zero but the model won't learn.")
    print()

    training_args = TrainingArguments(
        output_dir="./sft_sanity_check" if args.sanity else "./sft_checkpoint",
        per_device_train_batch_size=1 if args.sanity else 2,
        gradient_accumulation_steps=1 if args.sanity else 16,   # sanity: no accum for speed
        max_steps=args.max_steps if args.max_steps is not None else (20 if args.sanity else -1),
        num_train_epochs=1 if args.sanity else (0 if args.max_steps is not None else 2),
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1 if args.sanity else 10,
        save_steps=500 if not args.sanity else 50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,         # saves ~40% VRAM
        dataloader_num_workers=0 if args.sanity else 4,
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