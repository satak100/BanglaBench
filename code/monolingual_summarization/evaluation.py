# -*- coding: utf-8 -*-
"""
Monolingual Summarization Evaluation Script with Command-line Arguments
"""

import os
import logging
import argparse
from datasets import load_dataset, load_from_disk
from together import Together
import cohere
from tqdm import tqdm
import time
from rouge_score import rouge_scorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

instruct_prompt = "Please write a one-sentence Bengali summary/TL;DR of the given Bengali article. The summary must not be longer than a sentence and must be in Bengali. Just return the summary without any preamble, quotations, or explanations."
def generate_content_together(client, input_text, model_name):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": instruct_prompt},
            {"role": "user", "content": input_text},
        ],
        model=model_name
    )
    return response.choices[0].message.content

def generate_content_aya(client, input_text, model_name):
    response = client.chat(
        model=model_name,
        message= instruct_prompt+ "\n\n" + input_text
    )
    return response.text

def extract_summary(input_text):
  if "\n\n" in input_text:
    return input_text.split("\n\n")[1]
  else:
    return input_text

scorer = rouge_scorer.RougeScorer(['rouge2', ], use_stemmer=True, lang="bengali")

def main(api_key, service_choice, model_name, dataset_range):  
    dataset = load_from_disk('truncated_xlsum')
    logging.info("Dataset loaded successfully.")

    rouge_scores = []

    if service_choice == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
        client = Together(api_key=api_key)
        logging.info("Evaluating using Together API.")
        for data in tqdm(dataset["test"].select(range(dataset_range))):
            input_text = data["text"]
            response = generate_content_together(client, input_text, model_name)
            response = extract_summary(response)
            target_text = data["summary"]
            rouge = scorer.score(target_text, response)['rouge2'].fmeasure
            rouge_scores.append(rouge)

    elif service_choice == "cohere":
        client = cohere.Client(api_key)
        logging.info("Evaluating using Cohere API.")
        for data in tqdm(dataset["test"].select(range(dataset_range))):
            input_text = data["text"]
            response = generate_content_aya(client, input_text, model_name)
            response = extract_summary(response)
            target_text = data["summary"]
            rouge = scorer.score(target_text, response)['rouge2'].fmeasure
            rouge_scores.append(rouge)
            time.sleep(1)  # To avoid rate limiting

    else:
        logging.error("Invalid service choice. Please choose 'together' or 'cohere'.")
        return

    avg_rouge_score = sum(rouge_scores) / len(rouge_scores)
    logging.info(f"Average ROUGE-2 score for {service_choice} model '{model_name}': {avg_rouge_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate monolingual summarization models using Together or Cohere APIs.")
    parser.add_argument("api_key", type=str, help="API key for the chosen service")
    parser.add_argument("service_choice", choices=["together", "cohere"], help="Service choice: 'together' or 'cohere'")
    parser.add_argument("model_name", type=str, help="Model name for the chosen service")
    parser.add_argument("dataset_range", type=int, help="The number of dataset items to infer on (max 1012 items)")
    args = parser.parse_args()

    main(args.api_key, args.service_choice, args.model_name, args.dataset_range)