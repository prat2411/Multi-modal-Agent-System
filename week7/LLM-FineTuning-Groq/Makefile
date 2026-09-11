#!/bin/bash

# Makefile for LLM Fine-Tuning Pipeline

.PHONY: help install dev clean train infer eval prepare test docs

help:
	@echo "LLM Fine-Tuning Pipeline Commands"
	@echo "=================================="
	@echo "make install       - Install dependencies"
	@echo "make dev           - Install dev dependencies"
	@echo "make prepare       - Prepare data for training"
	@echo "make train         - Train the model"
	@echo "make infer         - Run inference"
	@echo "make eval          - Evaluate and visualize results"
	@echo "make clean         - Clean up generated files"
	@echo "make test          - Run tests"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

prepare:
	python prepare_data.py

train:
	python train.py

infer:
	python inference.py

eval:
	python results.py

clean:
	rm -rf __pycache__
	rm -rf models/*
	rm -rf results/*.png
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

test:
	pytest tests/ -v

format:
	black *.py pricer/*.py

lint:
	flake8 *.py pricer/*.py --max-line-length=100

type-check:
	mypy *.py pricer/*.py --ignore-missing-imports

docs:
	@echo "Documentation is in README.md and MODEL_CARD.md"

all: install prepare train eval
	@echo "Complete pipeline finished!"

.DEFAULT_GOAL := help
