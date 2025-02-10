# Providing Reasoning Capabilities to a QWEN2.5-7B Model

## Overview
This project provides a script for training a transformer-based language model using the Hugging Face `transformers` library and `TRL` (Transformer Reinforcement Learning). The script loads a pre-trained model, tokenizes a dataset, and fine-tunes the model on the provided training data.

## Features
- Supports various transformer-based models (e.g., Qwen, Llama)
- Loads datasets from Hugging Face Hub
- Implements a data collator for completion-only language modeling
- Uses `SFTTrainer` for supervised fine-tuning
- Saves the trained model and tokenizer

## Requirements
Before running the script, ensure you have the following dependencies installed:

```bash
pip install torch transformers datasets trl
```

## Usage
### 1. Configuration
The script uses a dataclass `TrainingConfig` to store training parameters. These parameters include:

- `model_name`: Name of the pre-trained model (default: `Qwen/Qwen2.5-7B-Instruct`)
- `block_size`: Maximum sequence length for training (default: `32768`)
- `train_file_path`: Path to the training dataset (default: `simplescaling/s1K_tokenized`)
- `dagger`: Boolean flag for an optional setting (default: `False`)

### 2. Running the Training Script
To train the model, simply run:

```bash
python train.py
```

The script will:
1. Parse command-line arguments using `HfArgumentParser`
2. Load the specified pre-trained model
3. Load and split the dataset into training and evaluation sets
4. Tokenize and preprocess the data
5. Configure the trainer and start the fine-tuning process
6. Save the trained model and tokenizer

### 3. Model Saving
After training, the fine-tuned model and tokenizer are saved to the directory specified by `args.output_dir`.

## Notes
- The script automatically adjusts settings based on the model type (e.g., `Qwen`, `Llama`)
- Uses a special padding token that is model-specific
- Implements `DataCollatorForCompletionOnlyLM` to mask user instructions and only compute loss over assistant responses
- Uses `FSDP` for efficient loading when training large models (e.g., `70B` models)

## Logging
The script logs training configurations and progress using Python’s built-in `logging` module. Logs include details about:
- Training configuration
- Dataset processing
- Training progress

## Future Improvements
- Add support for more dataset formats
- Implement advanced training techniques such as LoRA or PEFT for efficient fine-tuning
- Provide pre-configured training scripts for different hardware setups

## License
This project is licensed under the MIT License.

## Acknowledgments
This script leverages the Hugging Face `transformers` and `TRL` libraries for model fine-tuning and training.