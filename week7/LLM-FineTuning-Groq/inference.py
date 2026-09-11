"""
Inference Script using Groq API
Uses fine-tuned LoRA adapter for price estimation
"""

import os
from dotenv import load_dotenv

# HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Project imports
import config
from pricer.evaluator import GroqPredictor, Evaluator
from util import Tester

load_dotenv(override=True)


def load_fine_tuned_model():
    """Load fine-tuned model with LoRA adapter"""
    print("Loading base model with quantization...")
    
    # Quantization config
    if config.QUANT_4_BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(base_model, config.HUB_MODEL_NAME)
    except Exception as e:
        print(f"Note: Could not load from hub: {e}")
        print("Using base model for inference")
        model = base_model
    
    print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")
    
    return model, tokenizer


def evaluate_with_groq(test_data, size=200):
    """Evaluate using Groq API"""
    print("\n" + "="*50)
    print("Evaluating with Groq API")
    print("="*50)
    
    # Create Groq predictor
    groq_predictor = GroqPredictor()
    
    # Evaluate
    evaluator = Evaluator(predictor=groq_predictor)
    results = evaluator.evaluate_batch(test_data[:size])
    
    # Print results
    print(f"\nResults (first {size} samples):")
    print(f"Mean Absolute Error: ${results['mean_absolute_error']:.2f}")
    print(f"Total Samples: {results['total_samples']}")
    
    # Visualize
    print("\nGenerating visualization...")
    tester = Tester(groq_predictor, test_data, 
                   title="Groq API Price Estimator", size=size)
    tester.run()
    
    return results


def interactive_pricing():
    """Interactive pricing mode"""
    print("\n" + "="*50)
    print("Interactive Pricing Mode")
    print("="*50)
    print("Enter item details to get price predictions")
    print("Type 'quit' to exit\n")
    
    groq_predictor = GroqPredictor()
    
    while True:
        title = input("Item Title: ").strip()
        if title.lower() == 'quit':
            break
            
        description = input("Item Description: ").strip()
        if not description:
            description = "No additional description provided"
        
        # Create prompt
        prompt = f"Estimate the price of this item:\nTitle: {title}\nDescription: {description}\n\nPrice: $"
        
        # Get prediction
        print("\nPredicting price...")
        prediction = groq_predictor({'prompt': prompt})
        print(f"Predicted Price: ${prediction}")
        print("-" * 50 + "\n")


def main():
    """Main inference function"""
    print("="*50)
    print("LLM Fine-Tuning Inference - Groq Integration")
    print("="*50)
    
    # Load test data
    from datasets import load_dataset
    print(f"\nLoading test dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    test_data = dataset['test']
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Option 1: Evaluate with Groq API
    print("\nChoose evaluation method:")
    print("1. Evaluate with Groq API")
    print("2. Interactive pricing")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        size = input("Enter number of samples to evaluate (default 200): ").strip()
        size = int(size) if size.isdigit() else 200
        evaluate_with_groq(test_data, size=size)
    elif choice == "2":
        interactive_pricing()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
