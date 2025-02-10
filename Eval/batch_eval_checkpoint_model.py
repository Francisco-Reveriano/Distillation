import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def query_qwen2_5_batch(prompts, model, tokenizer, max_new_tokens=4096, device="cuda"):
    """
    Processes a batch of prompts through the model and extracts the assistant's responses.

    Args:
        prompts (list of str): List of formatted prompts.
        model: The loaded causal language model (possibly wrapped in DataParallel).
        tokenizer: The corresponding tokenizer.
        max_new_tokens (int): Maximum number of tokens to generate.
        device (str): Device to run the model on.

    Returns:
        list of str: The assistant responses for each prompt.
    """
    # Tokenize the batch with proper padding, truncation, and attention masks.
    encoded_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,           # Pad to the longest sequence in the batch.
        truncation=True,        # Truncate sequences longer than max_length.
        max_length=1024         # Set an appropriate maximum length.
    )
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    # If the model is wrapped in DataParallel, use the underlying module.
    model_to_use = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        outputs = model_to_use.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )

    responses = []
    for output in outputs:
        # Decode the generated text (keeping special tokens for extraction)
        raw_output = tokenizer.decode(output, skip_special_tokens=False)
        # Extract text after the assistant marker
        assistant_part = raw_output.split("<|im_start|>assistant")[-1]
        assistant_response = assistant_part.split("<|im_end|>")[0].strip()
        responses.append(assistant_response)

    return responses


def eval_original_model():
    # Load the dataset from parquet
    df = pd.read_parquet("hf://datasets/hendrydong/gpqa_diamond/data/test-00000-of-00001.parquet")

    # Load model and tokenizer
    model_name = "../ckpts/s1_20250210_074814"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()  # Set the model to evaluation mode

    # If multiple GPUs are available, wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Define the system message and extract all problems
    system_message = "You are a helpful AI assistant. Succinctly answer the provided question."
    problems = df["problem"].tolist()

    batch_size = 8  # Adjust based on your GPU memory and model size
    responses = []

    # Process the dataset in batches
    for i in tqdm(range(0, len(problems), batch_size), desc="Generating Responses"):
        batch_problems = problems[i: i + batch_size]
        # Format each prompt using the ChatML format
        batch_prompts = [
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            for p in batch_problems
        ]
        batch_responses = query_qwen2_5_batch(
            batch_prompts, model, tokenizer, max_new_tokens=32768, device=device
        )
        responses.extend(batch_responses)

    # Store the responses in the DataFrame and save to an Excel file
    df["Original_Model"] = responses
    df.to_excel("Results_Checkpoint_Model.xlsx")


if __name__ == "__main__":
    eval_original_model()
