"""
Item class for managing product data and prompts
"""

from typing import List, Dict, Any, Tuple
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
import json


class Item:
    """Represents an item with title, description, and price"""
    
    def __init__(self, title: str, summary: str, price: float):
        """
        Initialize Item
        
        Args:
            title: Product title
            summary: Product description/summary
            price: Product price
        """
        self.title = title
        self.summary = summary
        self.price = price
        self.prompt = None
        self.completion = None

    def count_tokens(self, tokenizer) -> int:
        """Count tokens in summary"""
        return len(tokenizer.encode(self.summary))

    def count_prompt_tokens(self, tokenizer) -> int:
        """Count tokens in prompt and completion"""
        if self.prompt and self.completion:
            tokens = len(tokenizer.encode(self.prompt)) + len(tokenizer.encode(self.completion))
            return tokens
        return 0

    def make_prompts(self, tokenizer, cutoff: int = 256, is_training: bool = True) -> None:
        """
        Create prompt and completion for fine-tuning
        
        Args:
            tokenizer: HuggingFace tokenizer
            cutoff: Maximum tokens for summary
            is_training: If True, format for training; otherwise for inference
        """
        # Truncate summary if needed
        tokens = tokenizer.encode(self.summary)
        if len(tokens) > cutoff:
            # Truncate and re-decode
            truncated = tokenizer.decode(tokens[:cutoff])
            summary_text = truncated
        else:
            summary_text = self.summary

        # Format prompt and completion
        self.prompt = f"Estimate the price of this item:\nTitle: {self.title}\nDescription: {summary_text}\n\nPrice: $"
        
        if is_training:
            self.completion = f"{self.price:.2f}"
        else:
            self.completion = f"{self.price:.2f}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Item':
        """Create Item from dictionary"""
        return cls(
            title=data.get('title', ''),
            summary=data.get('summary', ''),
            price=float(data.get('price', 0))
        )

    @classmethod
    def from_hub(cls, dataset_name: str) -> Tuple[List['Item'], List['Item'], List['Item']]:
        """
        Load items from HuggingFace Hub
        
        Args:
            dataset_name: Dataset identifier on Hub (e.g., "username/dataset_name")
            
        Returns:
            Tuple of (train_items, val_items, test_items)
        """
        dataset = load_dataset(dataset_name)
        
        train_items = [cls.from_dict(item) for item in dataset['train']]
        val_items = [cls.from_dict(item) for item in dataset['validation']]
        test_items = [cls.from_dict(item) for item in dataset['test']]
        
        return train_items, val_items, test_items

    @classmethod
    def push_prompts_to_hub(cls, dataset_name: str, 
                           train: List['Item'], 
                           val: List['Item'], 
                           test: List['Item']) -> None:
        """
        Push item prompts to HuggingFace Hub
        
        Args:
            dataset_name: Target dataset name
            train: Training items
            val: Validation items
            test: Test items
        """
        def items_to_dataset_dict(items_list):
            return {
                'prompt': [item.prompt for item in items_list],
                'completion': [item.completion for item in items_list],
            }

        # Create dataset dict
        dataset_dict = DatasetDict({
            'train': dataset_to_huggingface(items_to_dataset_dict(train)),
            'validation': dataset_to_huggingface(items_to_dataset_dict(val)),
            'test': dataset_to_huggingface(items_to_dataset_dict(test)),
        })

        # Push to Hub
        dataset_dict.push_to_hub(dataset_name)
        print(f"Pushed dataset to {dataset_name}")

    def __repr__(self) -> str:
        return f"Item(title={self.title}, price=${self.price:.2f})"


def dataset_to_huggingface(data_dict):
    """Convert dictionary to HuggingFace Dataset"""
    from datasets import Dataset
    return Dataset.from_dict(data_dict)
