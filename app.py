import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from transformers import default_data_collator

# Disable Weights & Biases (W&B) logging
os.environ["WANDB_DISABLED"] = "true"

# Force GPU usage (will raise an error if not available)
if not torch.cuda.is_available():
    raise EnvironmentError("GPU not available. This code requires a GPU to run.")

device = torch.device("cuda")  # Only use GPU
print(f"Using device: {device}")

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)  # Move model to GPU

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Load dataset (adjust split size for larger datasets if needed)
dataset = load_dataset("tweet_eval", "emotion", split="train[:200]")  # Training data

# Tokenization function with labels for causal language modeling
def tokenize_function(examples):
    inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = inputs["input_ids"].copy()  # Labels should be the same as input_ids
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Load evaluation dataset (validation split)
eval_dataset = load_dataset("tweet_eval", "emotion", split="validation")

# Tokenize the evaluation dataset
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# Default collator for dynamic padding (more memory-efficient)
data_collator = default_data_collator

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # You can increase this if you want to fine-tune longer
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,  # Add batch size for evaluation
    save_steps=200,
    save_total_limit=1,
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",  # Disable logging to external services (e.g., WandB)
    evaluation_strategy="steps",  # Evaluate every few steps
    eval_steps=100,  # Evaluate every 100 steps
    load_best_model_at_end=True,  # Automatically load the best model
    metric_for_best_model="eval_loss",  # Use eval loss to determine best model
)

# Define Trainer with evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,  # Include evaluation dataset
    data_collator=data_collator,
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")

# Function to generate text with more advanced options
def generate_text(prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # Move input tensor to GPU
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,  # Control creativity with temperature (lower = less random)
        top_k=top_k,  # Top-k sampling
        top_p=top_p,  # Top-p (nucleus) sampling
        repetition_penalty=repetition_penalty,  # Penalize repeating tokens
        num_return_sequences=1,  # Number of sequences to generate
        no_repeat_ngram_size=2  # Avoid repeating bigrams
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "The car"
generated_text = generate_text(prompt, max_length=100, temperature=0.7, top_k=40, top_p=0.9)
print(f"Generated text: {generated_text}")
