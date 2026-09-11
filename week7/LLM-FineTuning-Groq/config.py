"""
Project Configuration
LLM Fine-Tuning with LoRA/QLoRA and Groq API Integration
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ============ HuggingFace Configuration ============
HF_TOKEN = os.getenv('HF_TOKEN', '')
HF_USER = os.getenv('HF_USER', 'ed-donner')

# ============ Base Model Configuration ============
BASE_MODEL = os.getenv('BASE_MODEL', 'meta-llama/Llama-3.2-3B')
PROJECT_NAME = os.getenv('PROJECT_NAME', 'price-estimator')
LITE_MODE = os.getenv('LITE_MODE', 'False').lower() == 'true'

# ============ Dataset Configuration ============
DATA_USER = os.getenv('DATA_USER', HF_USER)
DATASET_NAME = os.getenv('DATASET_NAME', 
    f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"
)

# ============ Quantization Configuration ============
QUANT_4_BIT = os.getenv('QUANT_4_BIT', 'True').lower() == 'true'

# ============ Device Configuration ============
DEVICE = os.getenv('DEVICE', 'cuda')

# ============ Training Configuration ============
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0002'))
EPOCHS = int(os.getenv('EPOCHS', '3'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4'))

# ============ Groq API Configuration ============
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '512'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))

# ============ Paths ============
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============ Model Hub Configuration ============
RUN_NAME = "lora-finetuned-groq"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_NAME}-{RUN_NAME}"

# ============ LoRA Configuration ============
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ============ Validation Configuration ============
DEFAULT_EVAL_SIZE = 200
