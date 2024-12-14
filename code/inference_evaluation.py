# -*- coding: utf-8 -*-
"""
Inference Evaluation Script with Command-line Arguments
"""

import os
import logging
import argparse
from datasets import load_dataset
from together import Together
import cohere
from tqdm import tqdm
import time
import re
from utils import generate_content_together, generate_content_aya

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

instruct_prompt = """You will be given two sentences. Please determine whether the first sentence entails, contradicts, or is neutral to the second. Pay close attention to each word as you analyze the relation between the two sentences. Respond in the format: 
Thought: {thought on if the first second entails, contradicts, or is neutral to the second sentence}\n\nVerdict: {any one of <entailment>, <contradiction> or <neutral> tags}"""

def main(api_key, service_choice, model_name, dataset_range):  
    dataset = load_dataset("csebuetnlp/xnli_bn")
    logging.info("Dataset loaded successfully.")

    scores = []

    if service_choice == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
        client = Together(api_key=api_key)
        logging.info("Evaluating using Together API.")
        for data in tqdm(dataset["test"].select(range(dataset_range))):
            input_text = "Sentence 1 : " + data["sentence1"] + "\n\nSentence 2: " + data['sentence2']
            response = generate_content_together(client, instruct_prompt, input_text, model_name).split()[-1]
            if bool(re.search(r"contradiction", response, re.IGNORECASE)):
              response = 0
            elif bool(re.search(r"entailment", response, re.IGNORECASE)):
              response = 1
            elif bool(re.search(r"neutral", response, re.IGNORECASE)):
              response = 2
            else:
              response = -1
            target = data["label"]
            scores.append(1 if response == target else 0)

    elif service_choice == "cohere":
        client = cohere.Client(api_key)
        logging.info("Evaluating using Cohere API.")
        for data in tqdm(dataset["test"].select(range(dataset_range))):
            input_text = "Sentence 1 : " + data["sentence1"] + "\n\nSentence 2: " + data['sentence2']
            response = generate_content_aya(client, instruct_prompt, input_text, model_name).split()[-1]
            if bool(re.search(r"contradiction", response, re.IGNORECASE)):
              response = 0
            elif bool(re.search(r"entailment", response, re.IGNORECASE)):
              response = 1
            elif bool(re.search(r"neutral", response, re.IGNORECASE)):
              response = 2
            else:
              response = -1
            target = data["label"]
            scores.append(1 if response == target else 0)
            time.sleep(1)  # To avoid rate limiting

    else:
        logging.error("Invalid service choice. Please choose 'together' or 'cohere'.")
        return

    avg_score = sum(scores) / len(scores)
    logging.info(f"Average score for {service_choice} model '{model_name}': {avg_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate inference models using Together or Cohere APIs.")
    parser.add_argument("api_key", type=str, help="API key for the chosen service")
    parser.add_argument("service_choice", choices=["together", "cohere"], help="Service choice: 'together' or 'cohere'")
    parser.add_argument("model_name", type=str, help="Model name for the chosen service")
    parser.add_argument("dataset_range", type=int, help="The number of dataset items to infer on (max 4.9k items)")
    args = parser.parse_args()

    main(args.api_key, args.service_choice, args.model_name, args.dataset_range)