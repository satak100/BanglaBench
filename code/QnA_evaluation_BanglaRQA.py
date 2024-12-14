# -*- coding: utf-8 -*-
"""
Question-Answering Evaluation Script with Command-line Arguments
"""

import os
import logging
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from together import Together
from normalizer import normalize
from evaluate import load as load_metric
from utils import generate_content_together

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# System prompt for Together API
llama3system = """The user will provide a context and a question, both in Bengali. Read the context and the question carefully.
Respond with a JSON object with the following keys:

"answerable" (boolean, Is the question answerable from the context?)
"question_type" (yes-no / single-span / multiple-span)
"answer" ('হ্যাঁ' or 'না' for yes-no questions)/substring of the context for single-span/list of substrings of the multiple-span/'<NOT_IN_CONTEXT>')
"""

def map_type(example):
    type_map = {
        "causal": "single-span",
        "confirmation": "yes-no",
        "factoid": "single-span",
        "list": "multiple-span"
    }
    example["question_type"] = type_map[example["question_type"]]
    return example

def extract_json(response):
    try:
        response_json = response.split("}")[0] + "}"
        response_json = "{" + response_json.split("{")[1]
        return json.loads(response_json)
    except Exception as e:
        logging.error(f"Failed to extract JSON: {e}")
        return {
            "answerable": False,
            "question_type": "extractive",
            "answer": "<NOT_IN_CONTEXT>"
        }

def evaluate_responses(dataset, answers):
    predictions = []
    references = []
    
    for i in range(len(dataset["test"])):
        data = dataset["test"][i]["answers"]
        data["text"] = [normalize(x) for x in data["answer_text"]]
        data["answer_start"] = list(range(len(data["text"])))

        if int(dataset["test"][i]["is_answerable"]) == 0:
            data["text"] = []
            data["answer_start"] = []

        del data["answer_text"]
        del data["answer_type"]

        references.append({
            "answers": data,
            "id": str(i)
        })

        answer_data = answers[i]
        a1 = answer_data["answerable"]
        a2 = answer_data["answer"]
        if isinstance(a2, list):
            a2 = "; ".join([x.strip() for x in a2 if x.strip() != "ইত্যাদি"])

        predictions.append({
            "prediction_text": normalize(a2) if a1 else "",
            "id": str(i),
            "no_answer_probability": 0.0 if a1 else 1.0
        })
    
    squad_v2_metric = load_metric("squad_v2")
    return squad_v2_metric.compute(predictions=predictions, references=references)

def main(api_key, model_name):
    os.environ["TOGETHER_API_KEY"] = api_key
    client = Together(api_key=api_key)

    dataset = load_dataset("sartajekram/BanglaRQA")
    dataset = dataset.map(map_type)

    template = "Context:\n\n{}\n\nQuestion:\n\n{}"
    answers = []

    for data in tqdm(dataset["test"], desc="Generating answers"):
        input_text = template.format(data["context"], data["question_text"])
        response = generate_content_together(client, llama3system, input_text, model_name)

        try:
            extracted_response = extract_json(response)
        except Exception:
            logging.warning(f"Retrying failed response: {response}")
            response = generate_content_together(client, llama3system, input_text, model_name)
            extracted_response = extract_json(response)

        answers.append(extracted_response)

    results = evaluate_responses(dataset, answers)
    logging.info(f"Evaluation results: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bengali question-answering models using Together API.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for Together API")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for Together API")
    args = parser.parse_args()

    main(args.api_key, args.model_name)
