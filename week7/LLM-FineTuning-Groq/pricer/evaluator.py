"""
Model evaluation and inference using Groq API
"""

from typing import Dict, Any, List
from groq import Groq
import re
import os
from config import GROQ_API_KEY, GROQ_MODEL, MAX_TOKENS, TEMPERATURE


class GroqPredictor:
    """Uses Groq API to make price predictions"""
    
    def __init__(self, api_key: str = None, model: str = GROQ_MODEL, 
                 temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS):
        """
        Initialize Groq predictor
        
        Args:
            api_key: Groq API key
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or GROQ_API_KEY or os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def predict(self, item: Dict[str, Any]) -> str:
        """
        Make price prediction for an item
        
        Args:
            item: Dictionary with 'prompt' key containing the price estimation prompt
            
        Returns:
            Predicted price as string
        """
        if 'prompt' not in item:
            raise ValueError("Item must have 'prompt' key")

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": item['prompt']
                    }
                ]
            )
            
            response = message.content[0].text
            return response
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return "0"

    def __call__(self, item: Dict[str, Any]) -> str:
        """Make prediction callable"""
        return self.predict(item)


class Evaluator:
    """Evaluates model predictions"""
    
    def __init__(self, predictor=None):
        """
        Initialize Evaluator
        
        Args:
            predictor: Function or callable that makes predictions
        """
        self.predictor = predictor or GroqPredictor()

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate single item
        
        Args:
            item: Item with prompt and completion fields
            
        Returns:
            Dictionary with prediction and error metrics
        """
        prediction = self.predictor(item)
        
        # Post-process prediction
        try:
            predicted_price = self._parse_price(prediction)
        except:
            predicted_price = 0.0

        actual_price = float(item.get('completion', 0))
        error = abs(predicted_price - actual_price)
        error_percentage = (error / actual_price * 100) if actual_price > 0 else 0

        return {
            'prediction': predicted_price,
            'actual': actual_price,
            'error': error,
            'error_percentage': error_percentage,
            'raw_response': prediction
        }

    def evaluate_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate batch of items
        
        Args:
            items: List of items to evaluate
            
        Returns:
            Dictionary with aggregated metrics
        """
        results = []
        predictions = []
        actuals = []
        errors = []

        for item in items:
            result = self.evaluate_single(item)
            results.append(result)
            predictions.append(result['prediction'])
            actuals.append(result['actual'])
            errors.append(result['error'])

        return {
            'results': results,
            'mean_absolute_error': sum(errors) / len(errors) if errors else 0,
            'total_samples': len(items),
            'predictions': predictions,
            'actuals': actuals,
            'errors': errors
        }

    @staticmethod
    def _parse_price(response: str) -> float:
        """
        Extract price from model response
        
        Args:
            response: Model response string
            
        Returns:
            Parsed price as float
        """
        # Remove common currency symbols and formatting
        response = response.replace('$', '').replace(',', '')
        
        # Extract first number (int or float)
        match = re.search(r'[-+]?\d*\.?\d+', response)
        
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Could not parse price from response: {response}")
