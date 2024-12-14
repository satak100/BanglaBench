import sacrebleu

def generate_content_together(client, instruct_prompt, input_text, model_name):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": instruct_prompt},
            {"role": "user", "content": input_text},
        ],
        model=model_name
    )
    return response.choices[0].message.content

def generate_content_aya(client, instruct_prompt, input_text, model_name):
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

def extract_summary(input_text):
  if "\n\n" in input_text:
    return input_text.split("\n\n")[1]
  else:
    return input_text