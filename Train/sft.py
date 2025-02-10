import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

import torch
from datasets import load_dataset
import transformers
import trl
torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    block_size: int = 32768
    train_file_path: Optional[str] = 'simplescaling/s1K_tokenized'
    dagger: bool = False


def train():
    # Parse command-line arguments using HfArgumentParser.
    # This will also pick up the accelerator / distributed training parameters.
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses(
        look_for_args_file=False, return_remaining_strings=False)

    # Only the main process should log the training configuration
    if args.local_rank in (-1, 0):
        logging.info(f"Training config: { {**asdict(config), **asdict(args)} }")

    # For distributed training using TorchRun, set the CUDA device early.
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)

    # Load the model with special settings for very large (70B) models.
    if "70B" in config.model_name:
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    # Load the dataset and split the training data into train and eval sets.
    dataset = load_dataset(config.train_file_path)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Setup tokenizer and choose instruction/response templates based on the model.
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Set an unused token as pad token
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"
    else:
        # Fallback templates (adjust as needed)
        instruction_template = ""
        response_template = ""

    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Add or override some of the SFTConfig parameters via args
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size

    # (Optionally) move model to the proper device if not already handled by the accelerator.
    if args.local_rank != -1:
        model.to(torch.device("cuda", args.local_rank))

    # Instantiate the SFTTrainer using the proper train and eval datasets.
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
