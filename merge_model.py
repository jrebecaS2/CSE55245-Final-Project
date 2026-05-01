import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tinker_cookbook.weights import download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter into a full generative model checkpoint."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model identifier or local path for the pretrained model.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Local path to the PEFT adapter directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--tinker-download-path",
        default=None,
        help="Optional Tinker URI to download adapter weights before merging.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Torch dtype to use when loading the base model.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()

    if args.tinker_download_path is not None:
        print(f"Downloading adapter weights from Tinker: {args.tinker_download_path}")
        download(tinker_path=args.tinker_download_path, output_dir=args.adapter)

    print(f"Loading tokenizer from base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading base model from: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=resolve_dtype(args.torch_dtype),
    )

    print(f"Loading PEFT adapter from: {args.adapter}")
    peft_model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into the base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model and tokenizer to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Merge complete.")


if __name__ == "__main__":
    main()
