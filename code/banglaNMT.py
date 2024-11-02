# -*- coding: utf-8 -*-
"""
Translation Evaluation Script with Command-line Arguments
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

instruct_prompt = "You are a state-of-the-art AI assistant that translates sentences from Bengali to English. Just return the translation without any preamble, quotations or explanations."
# Define the template for generating translations with Together API
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

def calculate_sacrebleu(reference_sentence, candidate_sentence):
    reference = [[reference_sentence]]
    candidate = [candidate_sentence]
    sbleu = sacrebleu.corpus_bleu(candidate, reference)
    return sbleu.score

def main(api_key, service_choice, model_name):  
    dataset = load_dataset("csebuetnlp/BanglaNMT")
    logging.info("Dataset loaded successfully.")

    bleu_scores = []

    if service_choice == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
        client = Together(api_key=api_key)
        logging.info("Evaluating using Together API.")
        for data in tqdm(dataset["test"]):
            input_text = data["bn"]
            response = generate_content_together(client, input_text, model_name)
            target_text = data["en"]
            sbleu = calculate_sacrebleu(target_text, response)
            bleu_scores.append(sbleu)

    elif service_choice == "cohere":
        client = cohere.Client(api_key)
        logging.info("Evaluating using Cohere API.")
        for data in tqdm(dataset["test"]):
            input_text = data["bn"]
            response = generate_content_aya(client, input_text, model_name)
            target_text = data["en"]
            sbleu = calculate_sacrebleu(target_text, response)
            bleu_scores.append(sbleu)
            # time.sleep(1)  # To avoid rate limiting

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
    args = parser.parse_args()

    main(args.api_key, args.service_choice, args.model_name)
