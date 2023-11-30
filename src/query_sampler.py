import argparse
import numpy as np
import os
import requests
import sys
import spacy

from dotenv import load_dotenv, find_dotenv
from lingua import Language, LanguageDetectorBuilder

DATA_DIR = "../data"
HEADERS = { "Content-type": "application/json" }


def get_num_concepts(question, qminer_url):
    params = {      
        "query": question,
        "lang": "en",
    }               
    response = requests.get(qminer_url, params=params,
                            headers=HEADERS)
    return len(response.json()["concepts"])


def is_person_name(question, nlp):
    doc = nlp(question)
    for ent in doc.ents:
        if ent.label_ == "PER":
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-query-file", type=str, required=True,
                        help="Path to file containing gold queries")
    parser.add_argument("--query-log-file", type=str, required=True,
                        help="Path to file containing query log")
    parser.add_argument("--candidate-query-file", type=str, required=True,
                        help="Path to file containing candidate queries")
    args = parser.parse_args()

    gold_query_file = args.gold_query_file
    query_log_file = args.query_log_file
    candidate_query_file = args.candidate_query_file

    _ = load_dotenv(find_dotenv())

    qminer_url = os.environ["QMINER_URL"]

    token_lens, concept_lens = [], []
    with open(gold_query_file, "r") as fin:
        for line in fin:
            question = line.strip()
            num_tokens = len(question.split(" "))
            token_lens.append(num_tokens)
            num_concepts = get_num_concepts(question, qminer_url)
            concept_lens.append(num_concepts)

    token_lens = np.array(token_lens)
    mean_token_len = np.mean(token_lens)
    sd_token_len = np.std(token_lens)
    min_token_len = int(mean_token_len - sd_token_len)
    max_token_len = int(mean_token_len + sd_token_len + 1)

    concept_lens = np.array(concept_lens)
    mean_concept_len = np.mean(concept_lens)
    sd_concept_len = np.std(concept_lens)
    min_concept_len = int(mean_concept_len - sd_concept_len)
    max_concept_len = int(mean_concept_len + sd_concept_len + 1)

    langs = [Language.ENGLISH, Language.SPANISH]
    lang_detect = LanguageDetectorBuilder.from_languages(*langs).build()

    nlp = spacy.load("en_core_web_sm")

    already_seen = set()
    num_found = 0
    with open(query_log_file, "r") as fin, \
         open(candidate_query_file, "w") as fout:
        for line in fin:
            if line.startswith("SEARCH_QUERY"):
                continue
            try:
                query, timestamp = line.strip().split(",")
                num_tokens = len(query.split())
                if num_tokens >= min_token_len and num_tokens <= max_token_len:
                    num_concepts = get_num_concepts(query, qminer_url)
                    if (num_concepts >= min_concept_len and
                            num_concepts <= max_concept_len):
                        if query in already_seen:
                            continue
                        lang = lang_detect.detect_language_of(query)
                        if lang is None or lang != Language.ENGLISH:
                            continue
                        if is_person_name(query, nlp):
                            continue
                        print(num_found, query)
                        fout.write(query + "\n")
                        already_seen.add(query)
                        num_found += 1
            except ValueError:
                continue
            if num_found >= 500:
                break
