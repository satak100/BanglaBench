# -*- coding: utf-8 -*-
"""
Paraphrasing Evaluation Script with Command-line Arguments
"""

import os
import logging
import argparse
from datasets import load_dataset
from together import Together
import cohere
from tqdm import tqdm
import sacrebleu
import time
from utils import generate_content_together, generate_content_aya, calculate_sacrebleu

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

instruct_prompt = "You are a state-of-the-art AI assistant that generates Bengali paraphrases. The user provides you with a Bengali sentence, and your task is to generate a Bengali paraphrase of it. Just return the paraphrase without any preamble, quotations or explanations."

def main(api_key, service_choice, model_name, dataset_range):  
    dataset = load_dataset("csebuetnlp/BanglaParaphrase")
    logging.info("Dataset loaded successfully.")

    bleu_scores = []

    if service_choice == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
        client = Together(api_key=api_key)
        logging.info("Evaluating using Together API.")
        for data in tqdm(dataset["test"].select(range(dataset_range))):
            input_text = data["source"]
            response = generate_content_together(client, instruct_prompt, input_text, model_name)
            target_text = data["target"]
            sbleu = calculate_sacrebleu(target_text, response)
            bleu_scores.append(sbleu)

    elif service_choice == "cohere":
        client = cohere.Client(api_key)
        logging.info("Evaluating using Cohere API.")
        for data in tqdm(dataset["test"].select(range(dataset_range))):
            input_text = data["source"]
            response = generate_content_aya(client, instruct_prompt, input_text, model_name)
            target_text = data["target"]
            sbleu = calculate_sacrebleu(target_text, response)
            bleu_scores.append(sbleu)
            time.sleep(1)  # To avoid rate limiting

    else:
        logging.error("Invalid service choice. Please choose 'together' or 'cohere'.")
        return

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    logging.info(f"Average BLEU score for {service_choice} model '{model_name}': {avg_bleu_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translation models using Together or Cohere APIs.")
    parser.add_argument("api_key", type=str, help="API key for the chosen service")
    parser.add_argument("service_choice", choices=["together", "cohere"], help="Service choice: 'together' or 'cohere'")
    parser.add_argument("model_name", type=str, help="Model name for the chosen service")
    parser.add_argument("dataset_range", type=int, help="The number of dataset items to infer on (max 23k items)")
    args = parser.parse_args()

    main(args.api_key, args.service_choice, args.model_name, args.dataset_range)
