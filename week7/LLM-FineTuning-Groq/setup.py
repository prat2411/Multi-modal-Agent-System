"""Setup script for installation"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-finetuning-groq",
    version="1.0.0",
    author="Your Name",
    description="LLM Fine-Tuning Pipeline with Groq API Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "huggingface-hub>=0.19.0",
        "groq>=0.4.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "Flask>=3.0.0",
        "streamlit>=1.35.0",
    ],
)
