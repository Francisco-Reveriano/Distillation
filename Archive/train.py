import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set CUDA allocator configuration before any CUDA libraries are imported.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import trl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def prepare_dataset():
    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset = load_dataset("simplescaling/s1K_tokenized")
    train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return train_dataset, test_dataset, dataset

def load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.device_count() > 1:
        # Use device_map for model parallelism across GPUs
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = "cuda"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model.to(device)
    return tokenizer, model, device

def setup_collator(tokenizer):
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    # Use a token that is never used
    tokenizer.pad_token = "<|fim_pad|>"
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    return collator, tokenizer

# Let's test the base model before training
def format_qwen_prompt(system_message: str, user_message: str):
    """
    Formats the input prompt for Qwen2.5 models using ChatML format.

    Args:
        system_message (str): The system-level instruction.
        user_message (str): The user query.

    Returns:
        str: The formatted prompt.
    """
    prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def main():
    # Set the local device for distributed training only if a valid LOCAL_RANK is provided.
    if torch.cuda.is_available():
        local_rank_str = os.environ.get("LOCAL_RANK", None)
        if local_rank_str is not None:
            local_rank = int(local_rank_str)
            num_gpus = torch.cuda.device_count()
            print("Number of GPUs: ", num_gpus)
            if local_rank < num_gpus:
                torch.cuda.set_device(local_rank)
            else:
                print(
                    f"Warning: LOCAL_RANK {local_rank} is out of range (only {num_gpus} GPUs available). Defaulting to GPU 0.")
                torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(0)

    # Prepare the data
    train_dataset, test_dataset, dataset = prepare_dataset()

    # Load Model
    tokenizer, model, device = load_model(model_name="Qwen/Qwen2.5-7B-Instruct")

    # Setup Collator
    collator, tokenizer = setup_collator(tokenizer)

    # Configure training (using your existing SFTConfig settings)
    sft_config = SFTConfig(
        output_dir="../Model",
        max_steps=2000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=50,
        save_steps=5000,
        evaluation_strategy="steps",
        eval_steps=50,
        dataset_text_field="text",
        max_seq_length=4096,
        use_mps_device=(True if device == "mps" else False),
        fp16=True,
        optim="adamw_torch",
        logging_dir="./logs",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=test_dataset,
        args=sft_config,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Format with template
    system_message = "You are a helpful AI assistant."
    user_message = " Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?"
    formatted_prompt = format_qwen_prompt(system_message, user_message)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    print("Before training:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    trainer.train()

    system_message = "You are a helpful AI assistant."
    user_message = " Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?"
    formatted_prompt = format_qwen_prompt(system_message, user_message)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    print("After training:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    return model, tokenizer, device

if __name__ == "__main__":
    main()
