"""
LoRA/QLoRA Fine-Tuning Script with Groq API Integration
Fine-tunes Llama-3.2-3B on price estimation task
"""

import os
import torch
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Project imports
import config
from pricer.items import Item

load_dotenv(override=True)


def setup_model_and_tokenizer():
    """Load and configure base model with quantization"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Setup quantization
    if config.QUANT_4_BIT:
        print("Using 4-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
    else:
        print("Using 8-bit quantization...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pretraining_tp = 1
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def setup_lora(model):
    """Configure LoRA adapters"""
    print("Setting up LoRA adapters...")
    
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.LORA_TARGET_MODULES,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def prepare_data():
    """Load and prepare training data"""
    print(f"Loading dataset: {config.DATASET_NAME}")
    
    dataset = load_dataset(config.DATASET_NAME)
    
    print(f"Dataset splits: {dataset.keys()}")
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    return dataset


def format_prompts(examples):
    """Format prompts for training"""
    output_texts = []
    for prompt, completion in zip(examples['prompt'], examples['completion']):
        text = f"{prompt}{completion}"
        output_texts.append(text)
    return {"text": output_texts}


def train():
    """Main training function"""
    print("="*50)
    print("LLM Fine-Tuning with LoRA/QLoRA - Groq Integration")
    print("="*50)
    
    set_seed(42)
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    model = setup_lora(model)
    dataset = prepare_data()
    
    # Format data
    print("Formatting training data...")
    formatted_dataset = dataset.map(format_prompts, batched=True)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir=config.MODELS_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        report_to=[],  # Disable wandb
        bf16=torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.get_device_capability()[0] < 8,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=formatted_dataset['train'],
        eval_dataset=formatted_dataset['validation'],
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print("Saving model...")
    model.save_pretrained(os.path.join(config.MODELS_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.MODELS_DIR, "final_model"))
    
    # Push to hub
    print(f"Pushing to Hub: {config.HUB_MODEL_NAME}")
    try:
        model.push_to_hub(config.HUB_MODEL_NAME, private=False)
        tokenizer.push_to_hub(config.HUB_MODEL_NAME, private=False)
        print("✓ Successfully pushed to Hub")
    except Exception as e:
        print(f"Warning: Could not push to Hub: {e}")
    
    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")
    print("Training complete!")


if __name__ == "__main__":
    train()
