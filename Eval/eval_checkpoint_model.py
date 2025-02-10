import pandas as pd
from tqdm import tqdm
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

def query_qwen2_5(user_message: str, model, tokenizer) -> str:
    """
    Queries the Qwen2.5-14-Instruct model with the provided user_message
    and returns the assistant's response.

    Args:
        user_message (str): The user query.
        model: The loaded Qwen2.5 model.
        tokenizer: The Qwen2.5 tokenizer.

    Returns:
        str: The assistant's reply.
    """
    # Define the system message
    system_message = "You are a helpful AI assistant. Succinctly answer the provided question."

    # Format the prompt using ChatML format
    formatted_prompt = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Set the device and prepare inputs
    device = "cuda"
    encoded_inputs = tokenizer(formatted_prompt,
                               return_tensors="pt",
                               padding=False)
    inputs = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)
    model = model.to(device)

    # Generate the model output with a sufficient token budget and proper EOS handling
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=32768,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
    )

    # Decode the output (keeping the special tokens for extraction)
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract the assistant's response from the output
    assistant_part = raw_output.split("<|im_start|>assistant")[-1]
    assistant_response = assistant_part.split("<|im_end|>")[0].strip()

    return assistant_response

def eval_original_model():
    # Import Dataset
    df = pd.read_parquet("hf://datasets/hendrydong/gpqa_diamond/data/test-00000-of-00001.parquet")

    # Import Model
    model_name = "../ckpts/s1_20250210_074814"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Responses"):
        df.loc[index, "Original_Model"] = query_qwen2_5(df.loc[index, "problem"], model, tokenizer)
        df.to_excel("Results_Checkpoint_Model_Short.xlsx")


if __name__ == "__main__":
    eval_original_model()
