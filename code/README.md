# Directives
Running the evaluation scripts requires installing the necessary packages.
``` 
pip install -r requirements.txt 
```
Create an account in [Together AI](https://www.together.ai/) and [Cohere](https://cohere.com/) and get your API keys. 
List of models available in [Together](https://www.together.ai/models) and [Cohere](https://docs.cohere.com/v2/docs/models).

## Translation
Run the script `translation_evaluation.py` by specifying the API key, service choice (together or cohere) and model name.

```
python translation_evaluation.py your_api_key cohere c4ai-aya-expanse-32
```

## Monolingual Summarization
Directives available in `monolingual_summarization` directory

## Crosslingual Summarization
Run the script `summarization_evaluation.py` by specifying the API key, service choice and model name.

```
python summarization_evaluation.py your_api_key cohere c4ai-aya-expanse-32
```

## Paraphrase
Run the script `paraphrasing_evaluation.py` by specifying the API key, service choice, model name and dataset range (max 23k).

```
python paraphrasing_evaluation.py your_api_key cohere c4ai-aya-expanse-32 1000
```

## Inference
Run the script `inference_evaluation.py` by specifying the API key, service choice, model name and dataset range (max 4.9k).

```
python inference_evaluation.py your_api_key cohere c4ai-aya-expanse-32 1000
```

## Question Answering
Run the script `Qna_evaluation_BanglaRQA.py` by specifying the Together API key and model name.

```
python Qna_evaluation_BanglaRQA.py your_api_key meta-llama/Meta-Llama-3-70B-Instruct-Turbo
```
