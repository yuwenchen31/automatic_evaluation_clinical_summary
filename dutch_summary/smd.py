## Framework for Sentence Mover's Distance

import sys, nltk
import numpy as np
import spacy
import math
from wmd import WMD
from nltk.corpus import stopwords
from collections import Counter

# --- Global Setup ---
try:
    stop_words = set(stopwords.words('dutch'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('dutch'))

print("loading spacy: nl_core_news_lg")
# Note: nl_core_news_lg is used because it contains word vectors.
# If not installed, run: python -m spacy download nl_core_news_lg
nlp = spacy.load('nl_core_news_lg')

# --- Helper Functions (modified for single-pair processing) ---

def _tokenize_texts(ref_text, hyp_text, word_rep="glove"):
    """
    Tokenizes a single reference and hypothesis pair.
    """
    id_doc = []
    text_doc = []

    for text in [ref_text, hyp_text]:
        sent_list = [sent for sent in nltk.sent_tokenize(text)]
        if word_rep == "glove":
            IDs = [[nlp.vocab.strings[t.text.lower()] for t in nlp(sent) if t.text.isalpha() and t.text.lower() not in stop_words] for sent in sent_list]
        else: # Fallback for other representations, though not fully supported in this version
            IDs = [[nlp.vocab.strings[t.text] for t in nlp(sent)] for sent in sent_list]

        id_list = [x for x in IDs if x != []]
        text_list = [[token.text for token in nlp(x)] for x in sent_list if x != []]

        id_doc.append(id_list)
        text_doc.append(text_list)
        
    return id_doc, text_doc


def _get_embeddings(id_doc, text_doc, metric="sms", word_rep="glove"):
    """
    Gets embeddings for a single tokenized document pair.
    """
    rep_map = {}
    new_id = max(sum(sum(id_doc, []), [])) + 1 if sum(sum(id_doc, []), []) else 1
    sent_ids = [[], []]

    for i in range(2):
        for sent_i in range(len(id_doc[i])):
            word_emb_list = []
            if word_rep == "glove":
                for wordID in id_doc[i][sent_i]:
                    word_emb = nlp.vocab.get_vector(wordID)
                    word_emb_list.append(word_emb)

            if metric != "sms":
                for w_ind in range(len(word_emb_list)):
                    w_id = id_doc[i][sent_i][w_ind]
                    if w_id not in rep_map:
                        rep_map[w_id] = word_emb_list[w_ind]

            if (metric != "wms") and (len(word_emb_list) > 0):
                sent_emb = np.mean(np.array(word_emb_list), axis=0)
                rep_map[new_id] = sent_emb
                sent_ids[i].append(new_id)
                new_id += 1

    if metric != "wms":
        for j in range(len(id_doc)):
            id_doc[j].append(sent_ids[j])

    return id_doc, rep_map


def _get_weights(id_doc, metric="sms"):
    """
    Gets weights for a single document pair.
    """
    id_lists = [[], []]
    d_weights = [np.array([], dtype=np.float32), np.array([], dtype=np.float32)]

    for i in range(len(id_doc)):
        if metric != "wms":
            sent_ids = id_doc[i].pop()

        wordIDs = sum(id_doc[i], [])
        counts = Counter(wordIDs)

        if metric != "sms":
            for k in counts.keys():
                id_lists[i].append(k)
                d_weights[i] = np.append(d_weights[i], counts[k])

        if metric != "wms":
            id_lists[i] += sent_ids
            d_weights[i] = np.append(d_weights[i], np.array([float(len(x)) for x in id_doc[i] if x != []], dtype=np.float32))

    return id_lists, d_weights

# --- Main Calculation Function ---

def calculate_smd(reference_text, hypothesis_text, metric="sms", word_rep="glove"):
    """
    Calculates Sentence Mover's Distance (or other variants) for a single
    reference and hypothesis pair.

    :param reference_text: The reference text string.
    :param hypothesis_text: The hypothesis text string.
    :param metric: One of "sms", "wms", "s+wms". Defaults to "sms".
    :param word_rep: Embedding type. Defaults to "glove".
    :return: The similarity score.
    """
    if not isinstance(reference_text, str) or not isinstance(hypothesis_text, str) or not reference_text or not hypothesis_text:
        return 0.0

    try:
        token_doc, text_doc = _tokenize_texts(reference_text, hypothesis_text, word_rep)
        
        # If tokenization results in empty lists, cannot compute score
        if not token_doc[0] or not token_doc[1]:
            return 0.0

        [ref_ids, hyp_ids], rep_map = _get_embeddings(token_doc, text_doc, metric, word_rep)
        [ref_id_list, hyp_id_list], [ref_d, hyp_d] = _get_weights([ref_ids, hyp_ids], metric)

        # If there are no embeddings to compare, return 0
        if len(ref_id_list) == 0 or len(hyp_id_list) == 0:
            return 0.0

        doc_dict = {"0": ("ref", ref_id_list, ref_d), "1": ("hyp", hyp_id_list, hyp_d)}
        calc = WMD(rep_map, doc_dict, vocabulary_min=1)
        
        dist = calc.nearest_neighbors(str(0), k=1, early_stop=1)[0][1]
        sim = math.exp(-dist)
        
        return sim
    except Exception as e:
        print(f"Error calculating SMD: {e}")
        return 0.0