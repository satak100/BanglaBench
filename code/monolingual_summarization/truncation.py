# -*- coding: utf-8 -*-
"""
Dataset truncation Script with Command-line Arguments
"""

import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
UPPER_LIMIT = 8000
def truncate_long_articles(example):
    tokens = tokenizer(example["text"])["input_ids"]
    if len(tokens) > UPPER_LIMIT:
        example["text"] = tokenizer.decode(tokens[:UPPER_LIMIT], skip_special_tokens = True)
    return example

def main():  
    dataset = load_dataset("csebuetnlp/xlsum", 'bengali')
    logging.info("Dataset loaded successfully.")
    dataset["test"] = dataset["test"].map(truncate_long_articles)
    dataset.save_to_disk("truncated_xlsum")

if __name__ == "__main__":
    main()