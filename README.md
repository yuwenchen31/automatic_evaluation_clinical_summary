# automatic_evaluation_clinical_summary
Source code for automatic evaluation of LLM-generated clinical summary.

### Install Dependencies
To install all the packages used in this project, please install the dependencies by

```
pip install -r requirements.txt
```
### Files and Directories

- `main.py`

  
### Data 
The Dutch Clinical Discharge Summary dataset cannot be made available because it contains private patient data. Hallucinations-Generated-DI is available from [PhysioNet](https://physionet.org/content/ann-pt-summ/1.0.1/). 

### Model 
Models used for metric calculation. Please refer to individual website for 
Metric  | NL | EN |
-------------- | ---- | ---- |
SentenceBERT |	BioLORD-2023-M-Dutch-InContext-v1	| Bio_ClinicalBERT |
BERTscore |	BioLORD-2023-M-Dutch-InContext-v1	| roberta-large-mnli |
Perplexity | EuroLLM-1.7B	| EuroLLM-1.7b |
Sentence mover similarity (SMS)	| nl_core_news_lg	| en_core_web_lg |
word mover’s distance (WMD) |	fasttext model: nl.en.300.vec	| fasttext model: cc.en.300.vec |
WEEM4TS | fasttext model: nl.en.300.vec	| fasttext model: cc.en.300.vec |
MoverScore	| BioLORD-2023-M-Dutch-InContext-v1	| bert-base-multilingual-cased |
Semantic Coherence | bert-base-multilingual-cased | bert-base-multilingual-cased |
MTLD | nl_core_news_lg | en_core_web_lg |


Model download: 
[BioLORD-2023-M-Dutch-InContext-v1](https://huggingface.co/FremyCompany/BioLORD-2023-M-Dutch-InContext-v1)

[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

[roberta-large-mnli](https://huggingface.co/FacebookAI/roberta-large-mnli)

[EuroLLM-1.7b](http://huggingface.co/utter-project/EuroLLM-1.7B)

[bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) 

[fasttext model](https://fasttext.cc/docs/en/crawl-vectors.html) 

[Spacy model](https://spacy.io/models)
