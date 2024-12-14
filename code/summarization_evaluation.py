# -*- coding: utf-8 -*-
"""
Summary Evaluation Script with Command Line Arguments
"""


import os
import logging
import argparse
from datasets import load_dataset
from together import Together
import cohere
from transformers import AutoTokenizer
from tqdm import tqdm
from rouge_score import rouge_scorer
from utils import generate_content_together, generate_content_aya, extract_summary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

instruct_prompt = "Please write a one sentence Bengali summary/TL;DR of the given English article. The summary must be very concise and must be in Bangla. Just return the summary without any preamble, quotations, or explanations."

UPPER_LIMIT = 8000

def truncate_long_articles(example, tokenizer):
    tokens = tokenizer(example["text"])["input_ids"]
    if len(tokens) > UPPER_LIMIT:
        example["text"] = tokenizer.decode(tokens[:UPPER_LIMIT], skip_special_tokens=True)
    return example

def main(api_key, service_choice, model_name):
    ds = load_dataset("csebuetnlp/CrossSum", "english-bengali")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if service_choice == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
        client = Together(api_key=api_key)
        generate_content = lambda input_text: generate_content_together(client, instruct_prompt, input_text, model_name)
    elif service_choice == "cohere":
        client = cohere.Client(api_key)
        generate_content = lambda input_text: generate_content_aya(client, instruct_prompt, input_text, model_name)
    else:
        logging.error("Invalid service choice. Please select either 'together' or 'cohere'.")
        return
    
    # Preprocess dataset to truncate long articles
    ds = ds.map(lambda x: truncate_long_articles(x, tokenizer))

    generated_summaries = []
    for data in tqdm(ds["test"], desc="Generating summaries"):
        input_text = data["text"]
        generated_summary = generate_content(input_text)
        generated_summaries.append(generated_summary)

    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True, lang="bengali")

    # Calculate ROUGE-2 scores
    rouge_scores = []
    for i, generated_summary in enumerate(generated_summaries):
        reference_summary = ds["test"][i]["summary"]
        rouge_score = scorer.score(reference_summary, generated_summary)['rouge2'].fmeasure
        rouge_scores.append(rouge_score)

    average_rouge2 = sum(rouge_scores) / len(rouge_scores)
    logging.info(f"Average ROUGE-2 score (raw): {average_rouge2}")

    generated_summaries2 = [extract_summary(summary) for summary in generated_summaries]

    rouge_scores = []
    for i, generated_summary in enumerate(generated_summaries2):
        reference_summary = ds["test"][i]["summary"]
        rouge_score = scorer.score(reference_summary, generated_summary)['rouge2'].fmeasure
        rouge_scores.append(rouge_score)

    average_rouge2 = sum(rouge_scores) / len(rouge_scores)
    logging.info(f"Average ROUGE-2 score (concise): {average_rouge2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Bengali summaries from English articles and evaluate using ROUGE-2.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the chosen service")
    parser.add_argument("--service_choice", choices=["together", "cohere"], required=True, help="Service choice: 'together' or 'cohere'")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for tokenizer and chosen API service")
    args = parser.parse_args()

    main(args.api_key, args.service_choice, args.model_name)
