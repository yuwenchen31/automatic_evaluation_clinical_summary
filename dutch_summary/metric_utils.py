
from rouge_score import rouge_scorer
import sacrebleu
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
import pandas as pd
from moverscore import word_mover_score
from collections import defaultdict
from typing import List, Union, Iterable
from itertools import zip_longest
import numpy as np
import torch
from evaluate import logging
from nltk.translate import meteor_score
from packaging import version
from collections import Counter
import sacrebleu
import spacy

bert_model_hf_path = '../models/BioLORD-2023-M-Dutch-InContext-v1'
eurollm_model_hf_path = '../models/eurollm-1.7b'
bert_model_multi_hf_path = '../models/bert-base-multilingual-cased'

## PREPROCESSING 
def preprocessing(text): 

    if not hasattr(preprocessing, 'spacy_tokenizer'):
        preprocessing.spacy_tokenizer = spacy.load("nl_core_news_lg")
    # remove stopwords and punctuation
    doc = preprocessing.spacy_tokenizer(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    # remove custom patterns like [leeftijd-1], [persoon-1], [ziekenhuis-1]
    lemmas = [t for t in lemmas if not any(sub in t for sub in ['leeftijd', 'persoon', 'ziekenhuis'])]

    # Join back into string
    processed_text = " ".join(lemmas)
    
    return processed_text



## METEOR 
# https://github.com/huggingface/evaluate/blob/main/metrics/meteor/meteor.py
def meteor(predictions, references, alpha=0.9, beta=3, gamma=0.5):
        multiple_refs = isinstance(references[0], list)

        # the version of METEOR in NLTK version 3.6.5 and earlier expect tokenized inputs
        if multiple_refs:
            scores = [
                meteor_score.meteor_score(
                    [word_tokenize(ref, language='dutch') for ref in refs],
                    word_tokenize(pred, language='dutch'),
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                for refs, pred in zip(references, predictions)
            ]
        else:
            scores = [
                meteor_score.single_meteor_score(
                    word_tokenize(ref, language='dutch'), word_tokenize(pred, language='dutch'), alpha=alpha, beta=beta, gamma=gamma
                )
                for ref, pred in zip(references, predictions)
            ]
        

        return {"meteor": np.mean(scores)}



## PERPLEXITY: https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py

def perplexity(
        predictions, model_id=eurollm_model_hf_path, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=1024, stride=1024
    ):

    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        device = "cuda" if device == "gpu" else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # assign pad token if needed
    if tokenizer.pad_token is None and batch_size > 1:
        #existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        existing_special_tokens = list(tokenizer.special_tokens_map.values())
        assert len(existing_special_tokens) > 0, "Need a special token for padding."
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
    from torch.nn import CrossEntropyLoss
    loss_fct = CrossEntropyLoss(reduction="none")
    ppls = []
    from tqdm.auto import tqdm
    for text in tqdm(predictions, desc="Computing perplexity"):
        # tokenize the full text (no truncation)
        encodings = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)
        attn_mask = encodings["attention_mask"].to(device)

        nll_sum = 0.0
        token_count = 0
        prev_end = 0
        text_len = input_ids.size(1)

        # sliding window over long text
        for start in range(0, text_len, stride):
            end = min(start + max_length, text_len)
            trg_len = end - prev_end  # only count new tokens

            input_ids_chunk = input_ids[:, start:end]
            attn_mask_chunk = attn_mask[:, start:end]
            labels = input_ids_chunk.clone()
            labels[:, :-trg_len] = -100  # ignore overlapping context

            # add BOS token for first chunk if requested
            if add_start_token and start == 0 and tokenizer.bos_token_id is not None:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * input_ids_chunk.size(0)).to(device)
                input_ids_chunk = torch.cat([bos_tokens_tensor, input_ids_chunk], dim=1)
                attn_mask_chunk = torch.cat([torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask_chunk], dim=1)
                labels = torch.cat([torch.full((labels.size(0), 1), -100, dtype=torch.int64).to(device), labels], dim=1)

            with torch.no_grad():
                out_logits = model(input_ids_chunk, attention_mask=attn_mask_chunk).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attn_mask_chunk[..., 1:].contiguous()

            # sum NLL over tokens
            nll = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask).sum()
            nll_sum += nll
            token_count += shift_attention_mask.sum()

            prev_end = end
            if end == text_len:
                break

        # compute perplexity like original function
        ppl = torch.exp(nll_sum / token_count)
        ppls.append(ppl.item())

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}




# def perplexity_hf(
#         predictions, model_id = gpt2_model_hf_path, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
#     ):

#         if device is not None:
#             assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
#             if device == "gpu":
#                 device = "cuda"
#         else:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         model = AutoModelForCausalLM.from_pretrained(model_id)
#         model = model.to(device)

#         tokenizer = AutoTokenizer.from_pretrained(model_id)

#         # if batch_size > 1 (which generally leads to padding being required), and
#         # if there is not an already assigned pad_token, assign an existing
#         # special token to also be the padding token
#         if tokenizer.pad_token is None and batch_size > 1:
#             existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
#             # check that the model already has at least one special token defined
#             assert (
#                 len(existing_special_tokens) > 0
#             ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
#             # assign one of the special tokens to also be the pad token
#             tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

#         if add_start_token and max_length:
#             # leave room for <BOS> token to be added:
#             assert (
#                 tokenizer.bos_token is not None
#             ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
#             max_tokenized_len = max_length - 1
#         else:
#             max_tokenized_len = max_length

#         encodings = tokenizer(
#             predictions,
#             add_special_tokens=False,
#             padding=True,
#             truncation=True if max_tokenized_len else False,
#             max_length=max_tokenized_len,
#             return_tensors="pt",
#             return_attention_mask=True,
#         ).to(device)

#         encoded_texts = encodings["input_ids"]
#         attn_masks = encodings["attention_mask"]

#         # check that each input is long enough:
#         if add_start_token:
#             assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
#         else:
#             assert torch.all(
#                 torch.ge(attn_masks.sum(1), 2)
#             ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

#         ppls = []
#         loss_fct = CrossEntropyLoss(reduction="none")

#         for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
#             end_index = min(start_index + batch_size, len(encoded_texts))
#             encoded_batch = encoded_texts[start_index:end_index]
#             attn_mask = attn_masks[start_index:end_index]

#             if add_start_token and tokenizer.bos_token_id is not None:
#                 bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
#                 encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
#                 attn_mask = torch.cat(
#                     [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
#                 )

#             labels = encoded_batch

#             with torch.no_grad():
#                 out_logits = model(encoded_batch, attention_mask=attn_mask).logits

#             shift_logits = out_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

#             perplexity_batch = torch.exp(
#                 (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
#                 / shift_attention_mask_batch.sum(1)
#             )

#             ppls += perplexity_batch.tolist()

#         return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


# code adapted from https://github.com/rishibommasani/SummarizationEvaluation/blob/master/process_eval.py

# REDUNDANCY
def redundancy(row):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    red1_sent_scores, red2_sent_scores, redL_sent_scores = [], [], []
    summary = row["gpt_preprocessed"]
    sentences = sent_tokenize(summary)
    # sentences = [[str(token).lower() for token in spacy_tokenizer(s)] for s in sentences]
    if len(sentences) <= 1:
        red1_output = 0
        red2_output = 0
        redL_output = 0
    else:
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)): # ROUGE is symmetric, so only do one of (a,b), (b,a)
                s1 = sentences[i]
                s2 = sentences[j]
                scores = scorer.score(s1, s2)
                red1_sent_scores.append(scores["rouge1"].fmeasure)
                red2_sent_scores.append(scores["rouge2"].fmeasure)
                redL_sent_scores.append(scores["rougeL"].fmeasure)
        red1_output = sum(red1_sent_scores) / len(red1_sent_scores)
        red2_output = sum(red2_sent_scores) / len(red2_sent_scores)
        redL_output = sum(redL_sent_scores) / len(redL_sent_scores)
    
    return pd.Series({
        "redundancy_r1": red1_output, 
        "redundancy_r2": red2_output, 
        "redundancy_rL": redL_output })

# SEMANTIC COHERENCE 

def semantic_coherence(data, col_name="gpt_preprocessed"):
    print("Computing Semantic Coherence, using raw text")
    from transformers import BertTokenizer, BertForNextSentencePrediction
    tokenizer = BertTokenizer.from_pretrained(bert_model_multi_hf_path)
    model = BertForNextSentencePrediction.from_pretrained(bert_model_multi_hf_path)

    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    output = []
    for summary in tqdm(data[col_name]):
        # summary = ex['summary']
        scores = []
        sentences = sent_tokenize(summary)
        if len(sentences) <= 1:
            output.append(1)
        else:
            numerator = 0
            denominator = len(sentences) - 1
            for i in range(len(sentences) - 1):
                prev = sentences[i]
                curr = sentences[i + 1]
                s = "[CLS] " + prev + " [SEP] " + curr + " [SEP]"
                tokenized_text = tokenizer.tokenize(s)
                boundary = tokenized_text.index('[SEP]')
                segment_ids = [0] * boundary + [1] * (len(tokenized_text) - boundary)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segment_ids])
                with torch.no_grad():
                    prediction = model(tokens_tensor, token_type_ids=segments_tensors)[0]
                prediction_sm = softmax(prediction)[0].tolist()
                if prediction_sm[0] > 0.5:
                    numerator += 1
            output.append(round(numerator / denominator, 5))
    print(len(data), len(output))
    # assert len(output) == len(data)
    #return sum(output) / len(output)
    return output # per summary there is a semantic coherence score

# ABSTRACTIVITY 
def abstractivity(row, density_df):
    import spacy
    if not hasattr(abstractivity, 'spacy_tokenizer'):
        abstractivity.spacy_tokenizer = spacy.load("nl_core_news_sm")
    summary_len = len(abstractivity.spacy_tokenizer(row["gpt_preprocessed"]))
    density = density_df.loc[row.name, 'density']
    if summary_len > 0:
        score = 1 - (density / summary_len)
    else:
        score = 0  # empty summaries will have abstractivity of 0

    return pd.Series({"abstractivity": score})


# FAITHFULNESS (adapted from https://github.com/SamLee-dedeboy/Awesum/blob/main/server/metrics/stylistic.py#L177)
# NER-overlap 
# class FaithfulnessEvaluator:
#     def __init__(self):
#         self.model_ner = NERInaccuracyPenalty()
#         # self.ranges = ranges
#         self.fh = FaithfulnessHelper()
#         # model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
#         # model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")
#     # def __init__(self, original_text, summary):
#     #     self.original_text = original_text
#     #     self.summary = summary

#     def default(self, original_text, summary):
#         return self.ner_overlap([original_text], [summary])

#     def ner_overlap(self, sources, generateds):
#         source_ents = [self.model_ner.extract_entities(self.fh.replace_punctuation_with_whitespace(text)) for text in sources]
#         generated_ents = [self.model_ner.extract_entities(self.fh.replace_punctuation_with_whitespace(text)) for text in generateds]
#         # similar_source_ents = self.fh.get_similar_entities(source_ents)
#         similar_generated_ents = self.fh.get_similar_entities(generated_ents)
#         # reduced_source_ents = self.replace_similar_entities(similar_source_ents,source_ents)
#         reduced_generated_ents = self.fh.replace_similar_entities(similar_generated_ents,generated_ents)
#         match_count,top_source_entities = self.fh.top_entities_match(source_ents,reduced_generated_ents,str(sources))
#         # print(source_ents)
#         # print(generated_ents)
#         # print("+=====================")
#         # for source_ent, generated_ent, source in zip(source_ents, generated_ents, sources):
#         #     overlaps = self.find_overlaps(source_ent, generated_ent, source)
#         #     score = len(overlaps)/len(source_ents) 
#         #     scores.append(score)
#         score = match_count/len(top_source_entities) if len(top_source_entities) > 0 else 0
#         return score
#         if self.ranges!=[]:
#             if score<=self.ranges[0][1]: faithfulness_bin = "bad"
#             elif self.ranges[1][0]<=score<self.ranges[1][1]: faithfulness_bin = "low"
#             elif self.ranges[2][0]<=score<self.ranges[2][1]: faithfulness_bin = "avg"
#             else: faithfulness_bin = "good"
#             return score,faithfulness_bin
#         else: return score
#         # return {"scores": scores, "source_ents": source_ents, "gen_ents": generated_ents, "new_ents": all_new_ents}

#     def find_overlaps(self, ent_list_old, ent_list_new, source_text):
#         model_ner = self.model_ner
#         source_text = source_text.lower()

#         ent_set = set([model_ner.clean_entity_text(e["text"]) for e in ent_list_old])
#         overlaps = []

#         for ent_new in ent_list_new:
#             raw_entity_lower = ent_new["text"].lower()
#             entity_text = model_ner.clean_entity_text(ent_new["text"])
#             if model_ner.common_ents_no_problem(entity_text): # The entity is too common and could added anywhere
#                 overlaps.append(ent_new)
#                 continue
#             if entity_text in ent_set or model_ner.singular(entity_text) in ent_set: # Exact match with some entity
#                 overlaps.append(ent_new)
#                 continue
#             if entity_text in source_text or model_ner.singular(entity_text).lower() in source_text or raw_entity_lower in source_text: # Sometimes the NER model won't tag the exact same thing in the original paragraph, but we can just do string matching
#                 overlaps.append(ent_new)
#                 continue
#             # Starting the entity-specific matching
#             if ent_new["type"] in ["DATE", "CARDINAL", "MONEY", "PERCENT"]:
#                 # For dates:
#                 # a subset match is allowed: "several months" -> "months", "only a few weeks" -> "a few weeks"
#                 quantifier_clean = model_ner.quantifier_cleaning(ent_new["text"])
#                 if model_ner.quantifier_matching(ent_new["text"],  ent_list_old):
#                 # if any([clean_string in ent_text2 for ent_text2 in ent_set]):
#                     overlaps.append(ent_new)
#                     continue
                
#                 if all([w in source_text for w in quantifier_clean]):
#                     # A bit more desperate: remove additional words, and check that what's left is in the original
#                     overlaps.append(ent_new)
#                     continue
#                 if ent_new["type"] == "CARDINAL":
#                     if raw_entity_lower in model_ner.string2digits and model_ner.string2digits[raw_entity_lower] in source_text:
#                         overlaps.append(ent_new)
#                         continue # They wrote "nineteen" instead of 19
#                     elif raw_entity_lower in model_ner.digits2string and model_ner.digits2string[raw_entity_lower] in source_text.replace(",", ""):
#                         overlaps.append(ent_new)
#                         continue # They wrote 19 instead of "nineteen"

#             if ent_new["type"] == "GPE":
#                 if entity_text+"n" in ent_set or entity_text[:-1] in ent_set:
#                     overlaps.append(ent_new)
#                     # If you say india instead of indian, or indian instead of india.
#                     # Definitely doesn't work with every country, could use a lookup table
#                     continue
#             if ent_new["type"] in ["ORG", "PERSON"]:
#                 # Saying a smaller thing is fine: Barack Obama -> Obama. University of California, Berkeley -> University of California
#                 if any([entity_text in ent_text2 for ent_text2 in ent_set]):
#                     overlaps.append(ent_new)
#                     continue
#         return overlaps
#         # return {"score": score, "new_ents": new_ents2, "gen_entities": ents2, "source_entities": ents1}

#     def summac(self):
#         return 0
#         # return summac_score(self.original_text, self.summary)

# MOVERSCORE

# codes adapted from: https://github.com/AIPHES/emnlp19-moverscore/blob/master/examples/example.py
def moverscore_sentence(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 
    
    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score

def moverscore_corpus(sys_stream: List[str],
                     ref_streams:Union[str, List[Iterable[str]]], trace=0):

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]

    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]

    fhs = [sys_stream] + ref_streams

    corpus_score = 0
    for lines in zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
            
        hypo, *refs = lines
        corpus_score += moverscore_sentence(hypo, refs, trace=0)
        
    corpus_score /= len(sys_stream)

    return corpus_score


# SARI
# adapted from https://huggingface.co/spaces/evaluate-metric/sari/blob/main/sari.py

def SARIngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter()
    for sgram, scount in sgramcounter.items():
        sgramcounter_rep[sgram] = scount * numref

    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter()
    for cgram, ccount in cgramcounter.items():
        cgramcounter_rep[cgram] = ccount * numref

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmpscore1 = 0
    keeptmpscore2 = 0
    for keepgram in keepgramcountergood_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        # Fix an alleged bug [2] in the keep score computation.
        # keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram]
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    #      a target exactly.
    keepscore_precision = 1
    keepscore_recall = 1
    if len(keepgramcounter_rep) > 0:
        keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
    if len(keepgramcounterall_rep) > 0:
        # Fix an alleged bug [2] in the keep score computation.
        # keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
        keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values())
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = 2 * keepscore_precision * keepscore_recall / (keepscore_precision + keepscore_recall)

    # DELETION
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    deltmpscore2 = 0
    for delgram in delgramcountergood_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
        deltmpscore2 += delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    delscore_precision = 1
    if len(delgramcounter_rep) > 0:
        delscore_precision = deltmpscore1 / len(delgramcounter_rep)

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    addtmpscore = 0
    for addgram in addgramcountergood:
        addtmpscore += 1

    # Define 0/0=1 instead of 0 to give higher scores for predictions that match
    # a target exactly.
    addscore_precision = 1
    addscore_recall = 1
    if len(addgramcounter) > 0:
        addscore_precision = addtmpscore / len(addgramcounter)
    if len(addgramcounterall) > 0:
        addscore_recall = addtmpscore / len(addgramcounterall)
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = 2 * addscore_precision * addscore_recall / (addscore_precision + addscore_recall)

    return (keepscore, delscore_precision, addscore)


def SARIsent(ssent, csent, rsents):
    numref = len(rsents)

    s1grams = ssent.split(" ")
    c1grams = csent.split(" ")
    s2grams = []
    c2grams = []
    s3grams = []
    c3grams = []
    s4grams = []
    c4grams = []

    r1gramslist = []
    r2gramslist = []
    r3gramslist = []
    r4gramslist = []
    for rsent in rsents:
        r1grams = rsent.split(" ")
        r2grams = []
        r3grams = []
        r4grams = []
        r1gramslist.append(r1grams)
        for i in range(0, len(r1grams) - 1):
            if i < len(r1grams) - 1:
                r2gram = r1grams[i] + " " + r1grams[i + 1]
                r2grams.append(r2gram)
            if i < len(r1grams) - 2:
                r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                r3grams.append(r3gram)
            if i < len(r1grams) - 3:
                r4gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3]
                r4grams.append(r4gram)
        r2gramslist.append(r2grams)
        r3gramslist.append(r3grams)
        r4gramslist.append(r4grams)

    for i in range(0, len(s1grams) - 1):
        if i < len(s1grams) - 1:
            s2gram = s1grams[i] + " " + s1grams[i + 1]
            s2grams.append(s2gram)
        if i < len(s1grams) - 2:
            s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
            s3grams.append(s3gram)
        if i < len(s1grams) - 3:
            s4gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3]
            s4grams.append(s4gram)

    for i in range(0, len(c1grams) - 1):
        if i < len(c1grams) - 1:
            c2gram = c1grams[i] + " " + c1grams[i + 1]
            c2grams.append(c2gram)
        if i < len(c1grams) - 2:
            c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
            c3grams.append(c3gram)
        if i < len(c1grams) - 3:
            c4gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3]
            c4grams.append(c4gram)

    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1gramslist, numref)
    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2gramslist, numref)
    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3gramslist, numref)
    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4gramslist, numref)
    avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
    avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
    avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
    return finalscore


def sari_normalize(sentence, lowercase: bool = True, tokenizer: str = "13a", return_str: bool = True):

    # Normalization is requried for the ASSET dataset (one of the primary
    # datasets in sentence simplification) to allow using space
    # to split the sentence. Even though Wiki-Auto and TURK datasets,
    # do not require normalization, we do it for consistency.
    # Code adapted from the EASSE library [1] written by the authors of the ASSET dataset.
    # [1] https://github.com/feralvam/easse/blob/580bba7e1378fc8289c663f864e0487188fe8067/easse/utils/preprocessing.py#L7

    if lowercase:
        sentence = sentence.lower()

    if tokenizer in ["13a", "intl"]:
        if version.parse(sacrebleu.__version__).major >= 2:
            normalized_sent = sacrebleu.metrics.bleu._get_tokenizer(tokenizer)()(sentence)
        else:
            normalized_sent = sacrebleu.TOKENIZERS[tokenizer]()(sentence)
    elif tokenizer == "moses":
        normalized_sent = sacremoses.MosesTokenizer().tokenize(sentence, return_str=True, escape=False)
    elif tokenizer == "penn":
        normalized_sent = sacremoses.MosesTokenizer().penn_tokenize(sentence, return_str=True)
    else:
        normalized_sent = sentence

    if not return_str:
        normalized_sent = normalized_sent.split()

    return normalized_sent


def sari(sources, predictions, references):

    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("Sources length must match predictions and references lengths.")
    sari_score = 0
    for src, pred, refs in zip(sources, predictions, references):
        sari_score += SARIsent(sari_normalize(src), sari_normalize(pred), [sari_normalize(sent) for sent in refs])
    sari_score = sari_score / len(predictions)
    return {"sari": 100 * sari_score}


# WEEM4TS
# adapted from https://github.com/TuluTilahun/Text-Summarization/blob/master/WEEM4TS.py

add=0.0
def weem4ts(ref, gen, model, embwords, lang='nl'): 
    if lang == 'nl': 
        refsummary = [word_tokenize(text, language='dutch') for text in sent_tokenize(ref, language='dutch')]
        syssummary = [word_tokenize(text, language='dutch') for text in sent_tokenize(gen, language='dutch')]
    
    elif lang == 'en': 
        refsummary = [word_tokenize(text, language='english') for text in sent_tokenize(ref, language='english')]
        syssummary = [word_tokenize(text, language='english') for text in sent_tokenize(gen, language='english')]

    for i in range(len(refsummary)):

        notmatchwordREF=np.setdiff1d(refsummary[i],syssummary[i])
        notmatchwordSYS=np.setdiff1d(syssummary[i],refsummary[i])
        notmatchwordREFemb=[]
        for m in range(len(notmatchwordREF)):
            if notmatchwordREF[m] in embwords:
                notmatchwordREFemb.append(notmatchwordREF[m])
                
        sentweight=0.0
        unigramrecall=0.0
        weight=0.0
        countbigram=0.0
            
        for n in range(len(syssummary[i])):
            
            if syssummary[i][n] in refsummary[i]:
                weight=1.0
                sentweight=sentweight+weight
                if syssummary[i][n] in syssummary[i] and syssummary[i][n] in refsummary[i]:
                    if n<(len(syssummary[i])-1) and syssummary[i].index(syssummary[i][n])+1<len(syssummary[i]) and refsummary[i].index(syssummary[i][n])+1<len(refsummary[i]):
                        if syssummary[i][syssummary[i].index(syssummary[i][n])+1].lower() == (refsummary[i][refsummary[i].index(syssummary[i][n])+1]).lower():
                            countbigram=countbigram+1
                    
            elif len(notmatchwordREFemb)>0 and syssummary[i][n] in embwords:
                result = [model.similarity(syssummary[i][n], word) for word in notmatchwordREFemb]
                weight=max(result)
                sentweight=sentweight+weight
                if syssummary[i][n] in syssummary[i] and notmatchwordREFemb[result.index(max(result))] in refsummary[i]:
                    if n<(len(syssummary[i])-1) and syssummary[i].index(syssummary[i][n])+1<len(syssummary[i]) and refsummary[i].index(notmatchwordREFemb[result.index(max(result))])+1<len(refsummary[i]):
                        if (syssummary[i][syssummary[i].index(syssummary[i][n])+1]).lower() == (refsummary[i][refsummary[i].index(notmatchwordREFemb[result.index(max(result))])+1]).lower():
                            countbigram=countbigram+1
            else:
                weight=0.0
                sentweight=sentweight+weight 
        
    
        if len(refsummary[i])>0 or len(syssummary[i])>0:
            unigramrecall=sentweight/max(len(refsummary[i]),len((syssummary[i])))  # this is for fair evaluation for system summaries that is too less or too high than reference summary
        else:
            unigramrecall=0
            
        bigramprecision=(countbigram/(len(syssummary[i])-1))*100
        unigramrecall=unigramrecall*100

        WEEM4TSscore=(0.8*unigramrecall)+(0.2*bigramprecision)   # alpha = 0/8, betta=0.2
       
        return WEEM4TSscore


print("Loaded metric_utils:", __file__)