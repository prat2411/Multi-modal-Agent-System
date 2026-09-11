"""
Data Preparation Script
Handles tokenization, prompt creation, and dataset uploading
"""

import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer
from tqdm.notebook import tqdm
from datasets import load_dataset

import config
from pricer.items import Item

load_dotenv(override=True)


def main():
    """Main data preparation workflow"""
    print("="*50)
    print("Data Preparation for LoRA Fine-Tuning")
    print("="*50)
    
    # Login to HuggingFace
    print("\nLogging into HuggingFace Hub...")
    hf_token = config.HF_TOKEN or os.getenv('HF_TOKEN')
    if hf_token:
        login(hf_token, add_to_git_credential=True)
    else:
        print("Warning: HF_TOKEN not set. Some features may not work.")
    
    # Load dataset
    print(f"\nLoading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get data splits
    if 'train' in dataset:
        train = [Item.from_dict(item) for item in dataset['train']]
    else:
        train = []
    
    if 'validation' in dataset:
        val = [Item.from_dict(item) for item in dataset['validation']]
    else:
        val = []
    
    if 'test' in dataset:
        test = [Item.from_dict(item) for item in dataset['test']]
    else:
        test = []
    
    items = train + val + test
    
    print(f"✓ Loaded {len(train):,} training items, "
          f"{len(val):,} validation items, "
          f"{len(test):,} test items")
    
    # Setup tokenizer
    print(f"\nUsing tokenizer from: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    
    # Analyze token distribution
    print("\n--- Token Analysis ---")
    print("Counting tokens in summaries...")
    token_counts = [item.count_tokens(tokenizer) for item in tqdm(items)]
    
    # Visualization
    plt.figure(figsize=(15, 6))
    plt.title(f"Tokens in Summary: Avg {sum(token_counts)/len(token_counts):,.1f} "
              f"and highest {max(token_counts):,}\n")
    plt.xlabel('Number of tokens in summary')
    plt.ylabel('Count')
    plt.hist(token_counts, rwidth=0.7, color="skyblue", bins=range(0, 200, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'token_distribution.png'))
    plt.show()
    
    # Determine cutoff
    CUTOFF = 110
    cut = len([count for count in token_counts if count > CUTOFF])
    print(f"\nWith CUTOFF={CUTOFF}, will truncate {cut:,} items ({cut/len(items):.1%})")
    
    # Create prompts
    print("\nCreating prompts and completions...")
    for item in tqdm(train + val):
        item.make_prompts(tokenizer, CUTOFF, is_training=True)
    for item in tqdm(test):
        item.make_prompts(tokenizer, CUTOFF, is_training=False)
    
    # Show example
    if test:
        print("\nExample prompt/completion pair:")
        print("PROMPT:")
        print(test[0].prompt)
        print("\nCOMPLETION:")
        print(test[0].completion)
    
    # Analyze prompt tokens
    print("\n--- Prompt Token Analysis ---")
    print("Counting tokens in prompts and completions...")
    prompt_token_counts = [item.count_prompt_tokens(tokenizer) for item in tqdm(items)]
    
    # Visualization
    plt.figure(figsize=(15, 6))
    plt.title(f"Total Tokens: Avg {sum(prompt_token_counts)/len(prompt_token_counts):,.1f} "
              f"and highest {max(prompt_token_counts):,}\n")
    plt.xlabel('Number of tokens (prompt + completion)')
    plt.ylabel('Count')
    plt.hist(prompt_token_counts, rwidth=0.7, color="gold", bins=range(0, 200, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'prompt_token_distribution.png'))
    plt.show()
    
    # Upload to Hub (optional)
    print("\nData preparation complete!")
    print(f"Results saved to: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
