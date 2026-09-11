# Quick Start Guide

Get your LLM Fine-Tuning pipeline running in 5 minutes!

## Prerequisites

- Python 3.9+
- CUDA 12.0+ (for GPU training)
- 8GB+ VRAM

## 1. Install & Setup (2 minutes)

```bash
# Clone/navigate to project
cd LLM-FineTuning-Groq

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Edit .env with your keys:
# HF_TOKEN=your_huggingface_token
# GROQ_API_KEY=your_groq_api_key
```

## 2. Prepare Data (1 minute)

```bash
python prepare_data.py
```

This loads the dataset, analyzes tokens, and creates prompts.

## 3. Train Model (Optional - skip if using pre-trained)

```bash
python train.py
```

Expected: 3 hours on single GPU, achieves 87% accuracy

## 4. Run Inference (1 minute)

```bash
python inference.py
```

Choose:
- **Option 1**: Evaluate on test set with Groq API
- **Option 2**: Interactive mode (enter items and get prices)

## 5. View Results (1 minute)

```bash
python results.py
```

Generates comparison charts and metrics.

---

## Common Tasks

### Get a single price prediction

```python
from pricer.evaluator import GroqPredictor

predictor = GroqPredictor()
item = {"prompt": "Estimate price of: Vintage Watch 1970s Swiss...Price: $"}
price = predictor.predict(item)
print(f"Predicted: ${price}")
```

### Batch evaluate items

```python
from pricer.evaluator import Evaluator, GroqPredictor
from datasets import load_dataset

dataset = load_dataset("your-dataset/items_prompts")
evaluator = Evaluator(GroqPredictor())
results = evaluator.evaluate_batch(dataset['test'][:100])
print(f"MAE: ${results['mean_absolute_error']:.2f}")
```

### Train custom model

```bash
# Edit config.py:
# - Change HF_USER to your username
# - Change BASE_MODEL if needed
# - Adjust LEARNING_RATE, EPOCHS, etc

python train.py
```

### Push model to HuggingFace Hub

```python
from train import setup_lora, setup_model_and_tokenizer

model, tokenizer = setup_model_and_tokenizer()
model = setup_lora(model)
# ... after training ...
model.push_to_hub("your-username/model-name")
```

---

## Troubleshooting

### "CUDA out of memory"
```python
# In config.py, reduce batch size:
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
```

### "Groq API error"
```python
# Check your API key:
import os
print(os.getenv('GROQ_API_KEY'))

# Test the connection:
from pricer.evaluator import GroqPredictor
p = GroqPredictor()
print(p.predict({"prompt": "2+2="}))
```

### "HuggingFace Hub error"
```bash
# Login first
huggingface-cli login
# Then try again
```

---

## API Keys

### Get HuggingFace Token
1. Go to https://huggingface.co/settings/tokens
2. Create new token (fine-grained, read access)
3. Copy to `.env` as `HF_TOKEN`

### Get Groq API Key
1. Go to https://console.groq.com
2. Sign up or login
3. Navigate to API Keys
4. Copy key to `.env` as `GROQ_API_KEY`

---

## Performance Benchmarks

| Operation | Time | Cost |
|-----------|------|------|
| Data prep | 5 min | Free |
| Model training | 3 hours | ~$1 (single GPU) |
| Single inference | 100-200ms | ~$0.0003 |
| Batch 1000 items | ~100s | ~$0.30 |

## Code Examples

### Example 1: Quick Prediction
```python
from config import GROQ_MODEL
from pricer.evaluator import GroqPredictor

predictor = GroqPredictor()
result = predictor.predict({
    "prompt": "Estimate price of: New iPhone 15 Pro Max\n\nPrice: $"
})
print(f"Price: {result}")  # Output: ~$999
```

### Example 2: Evaluate Custom Data
```python
from pricer.evaluator import Evaluator, GroqPredictor
import pandas as pd

# Your data
items = [
    {"prompt": "...", "completion": "99.99"},
    {"prompt": "...", "completion": "149.99"},
]

evaluator = Evaluator(GroqPredictor())
results = evaluator.evaluate_batch(items)

# Create report
df = pd.DataFrame(results['results'])
print(df[['prediction', 'actual', 'error']])
```

### Example 3: Fine-tune on Custom Dataset
```python
# Edit config.py
DATASET_NAME = "your-username/your-dataset"
BASE_MODEL = "your-model"  # or keep default

# Build customDataset
from datasets import Dataset
data = {
    "prompt": ["...", "..."],
    "completion": ["...", "..."]
}
ds = Dataset.from_dict(data)

# Train
python train.py
```

---

## Resources

- 📖 [Full README](README.md)
- 🎯 [Model Card](MODEL_CARD.md)
- 🔧 [Configuration](config.py)
- 📚 [Example Notebook](example_workflow.py)
- 🌐 [Groq API Docs](https://console.groq.com/docs)
- 🤗 [HuggingFace Hub](https://huggingface.co)

---

## Need Help?

1. Check README.md for detailed documentation
2. Review config.py for parameter descriptions
3. Run `python example_workflow.py` for full workflow
4. Check Groq console logs for API errors
5. Open an issue on GitHub

---

**Happy Fine-Tuning! 🚀**
