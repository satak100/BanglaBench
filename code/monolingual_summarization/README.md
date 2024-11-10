# Directives
Evaluating models on xlsum dataset needs some pre-processing due to the input size limit constraints. The dataset needs to be truncated before evaluation.

## Truncation
Run
```
python truncation.py
```
This should prepare the truncated_xlsum directory that will be used for evaluation.
## Evaluation
First, install ROUGE-2 scoring dependencies
```
git clone https://github.com/csebuetnlp/xl-sum.git
cd ./xl-sum/multilingual_rouge_scoring
pip3 install -r requirements.txt
pip3 install --upgrade ./
mv ./xl-sum/multilingual_rouge_scoring/tokenizers.py ./xl-sum/multilingual_rouge_scoring/rouge_tokenizers.py
```
Then you can run the evaluation.py script for evaluation.