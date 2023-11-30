# Feature selection for training LTR models
# Adapted from: https://www.microsoft.com/en-us/research/project/mslr/
#
import argparse
import json
import numpy as np
import os
import re
import requests
import shelve

from dataclasses import dataclass, asdict
from dotenv import load_dotenv, find_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Any, List, Set, Tuple

DATA_DIR = "../data"
HEADERS = { "Content-type": "application/json" }


@dataclass
class Doc:
    doc_id: str
    title: str
    section_title: str
    bread_crumb: str
    text: str
    content_type: str
    is_populated: bool = False
    title_tokens: List[str] = None
    section_title_tokens: List[str] = None
    bread_crumb_tokens: List[str] = None
    text_tokens: List[str] = None
    num_title_tokens: int = None
    num_section_title_tokens: int = None
    num_bread_crumb_tokens: int = None
    num_text_tokens: int = None
    num_title_token_overlap: int = None
    num_section_title_token_overlap: int = None
    num_bread_crumb_token_overlap: int = None
    num_text_token_overlap: int = None
    #
    title_concept_counts: Dict[str, int] = None
    section_title_concept_counts: Dict[str, int] = None
    bread_crumb_concept_counts: Dict[str, int] = None
    text_concept_counts: Dict[str, int] = None
    #
    title_stygroup_counts: Dict[str, int] = None
    section_title_stygroup_counts: Dict[str, int] = None
    bread_crumb_stygroup_counts: Dict[str, int] = None
    text_stygroup_counts: Dict[str, int] = None
    #
    title_concept_overlap: int = None
    section_title_concept_overlap: int = None
    bread_crumb_concept_overlap: int = None
    text_concept_overlap: int = None
    #
    title_stygroup_overlap: int = None
    section_title_stygroup_overlap: int = None
    bread_crumb_stygroup_overlap: int = None
    text_stygroup_overlap: int = None
    # tf, tfidf
    title_ttf: float = None
    title_min_tf: float = None
    title_max_tf: float = None
    title_mean_tf: float = None
    title_var_tf: float = None
    title_min_tfidf: float = None
    title_max_tfidf: float = None
    title_mean_tfidf: float = None
    title_var_tfidf: float = None
    section_title_ttf: float = None
    section_title_min_tf: float = None
    section_title_max_tf: float = None
    section_title_mean_tf: float = None
    section_title_var_tf: float = None
    section_title_min_tfidf: float = None
    section_title_max_tfidf: float = None
    section_title_mean_tfidf: float = None
    section_title_var_tfidf: float = None
    bread_crumb_ttf: float = None
    bread_crumb_min_tf: float = None
    bread_crumb_max_tf: float = None
    bread_crumb_mean_tf: float = None
    bread_crumb_var_tf: float = None
    bread_crumb_min_tfidf: float = None
    bread_crumb_max_tfidf: float = None
    bread_crumb_mean_tfidf: float = None
    bread_crumb_var_tfidf: float = None
    text_ttf: float = None
    text_min_tf: float = None
    text_max_tf: float = None
    text_mean_tf: float = None
    text_var_tf: float = None
    text_min_tfidf: float = None
    text_max_tfidf: float = None
    text_mean_tfidf: float = None
    text_var_tfidf: float = None
    title_bm25_score: float = None
    section_title_bm25_score: float = None
    bread_crumb_bm25_score: float = None
    text_bm25_score: float = None
    title_vec_score: float = None
    section_title_vec_score: float = None
    bread_crumb_vec_score: float = None
    text_vec_score: float = None


@dataclass
class Query:
    query: str
    query_tokens: List[str] = None
    num_query_tokens: int = None
    concept_counts: Dict[str, int] = None
    stygroup_counts: Dict[str, int] = None


# from: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def tokenize(s: str) -> List[str]:
    s = re.sub('([.,!?()])', r' \1 ', s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def qminer_annotate(s: str,
                    qminer_api: str,
                    filter_concepts: Set[str] = None,
                    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """ return concept and semantic group counts found for the given
        string. If filter_concepts is provided, only report counts
        for concepts (and semantic groups) associated with the concepts
        provided in the filter_concepts set.
    """
    params = {
        "query": s,
        "lang": "en"
    }
    response = requests.get(qminer_api, params=params,
                            headers=HEADERS)
    concept_counts, stygroup_counts = {}, {}
    for doc in response.json()["concepts"]:
        concept = doc["imuid"]
        if filter_concepts is not None and concept not in filter_concepts:
            continue
        if concept not in concept_counts.keys():
            concept_counts[concept] = 1
        else:
            concept_counts[concept] += 1
        for sty_info in doc["semantic_infos"]:
            sty_group = sty_info["stygrp"]
            if sty_group not in stygroup_counts.keys():
                stygroup_counts[sty_group] = 1
            else:
                stygroup_counts[sty_group] += 1
    return concept_counts, stygroup_counts


def compute_weighted_overlap(query_dict: Dict[str, int],
                             es_index: str,
                             doc_dict: Dict[str, int]) -> int:
    """ compute the numbe of concepts or sty groups found in the
        query that also occur in the document field, and weight 
        occurrences of these concepts by the product of the query
        and field count for that concept, then sum.
    """
    overlap = 0
    for key_q in query_dict.keys():
        for key_d in doc_dict.keys():
            if key_q == key_d:
                overlap += query_dict[key_q] * doc_dict[key_q]
    return overlap


def get_doc(doc_id: str, elastic: Elasticsearch,
            cache: shelve.DbfilenameShelf
            ) -> Dict[str, str]:
    if doc_id in cache.keys():
        return cache[doc_id]
    resp = elastic.search(index=es_index, query={
        "match": {
            "_id": doc_id
        }
    })
    doc = {}
    for hit in resp["hits"]["hits"]:
        doc = Doc(
            doc_id=hit["_id"],
            title=hit["_source"]["title"],
            section_title=hit["_source"]["section_title"],
            bread_crumb=hit["_source"]["bread_crumb"],
            text=hit["_source"]["text"],
            content_type=hit["_source"]["content_type"],
        )
        break
    cache[doc_id] = doc
    return doc


def compute_index_statistics(doc_id: str,
                             elastic: Elasticsearch,
                             es_index: str
                             ) -> Dict[str, Any]:
    fields = ["title", "section_title", "bread_crumb_text", "text"]
    resp = elastic.termvectors(index=es_index,
                               id=doc_id,
                               fields=fields,
                               term_statistics=True)
    index_stats = {}
    for field_key, field_info in resp["term_vectors"].items():
        if field_key == "bread_crumb_text":
            field_key = "bread_crumb"
        doc_count = field_info["field_statistics"]["doc_count"]
        tfs, idfs, tf_idfs = [], [], []
        for term, term_stats in field_info["terms"].items():
            tf = term_stats["term_freq"]
            idf = np.log(doc_count / term_stats["doc_freq"])
            tf_idf = tf * idf
            tfs.append(tf)
            idfs.append(idf)
            tf_idfs.append(tf_idf)
        index_stats[field_key] = {
            "sum_ttf": field_info["field_statistics"]["sum_ttf"],
            "min_tf": np.min(tfs),
            "max_tf": np.max(tfs),
            "mean_tf": np.mean(tfs),
            "var_tf": np.var(tfs),
            "min_tfidf": np.min(tf_idfs),
            "max_tfidf": np.max(tf_idfs),
            "mean_tfidf": np.mean(tf_idfs),
            "var_tfidf": np.var(tf_idfs)
        }
    return index_stats


def compute_bm25_score(query: str, doc: Doc,
                       field_names: List[str],
                       elastic: Elasticsearch,
                       es_index: str,
                       ) -> Dict[str, float]:
    field_scores = {}
    for field_name in field_names:
        query = {
            "bool": {
                "must": [
                    {
                        "match": {
                            field_name: getattr(doc, field_name)
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {
                            "_id": doc.doc_id
                        }
                    }
                ]
            }
        }
        resp = elastic.search(index=es_index, query=query)
        for hit in resp["hits"]["hits"]:
            field_scores[field_name] = hit["_score"]
            break
    return field_scores


def compute_vector_similarity(query: str, doc: Doc,
                              sts_encoder: SentenceTransformer
                              ) -> Dict[str, float]:
    fields = ["title", "section_title", "bread_crumb", "text"]
    field_values = [query]
    for field_name in fields:
        field_values.append(getattr(doc, field_name))
    field_encs = sts_encoder.encode(field_values,
                                    convert_to_numpy=True)
    query_encs = np.tile(field_encs[0], (len(fields), 1))
    field_encs = field_encs[1:]
    sims = util.cos_sim(query_encs, field_encs)
    vector_sims = {}
    for i, field_name in enumerate(fields):
        vector_sims[field_name] = sims[i][i].numpy().item()
    return vector_sims


def generate_query_features(query: Query) -> Query:
    # token counts
    query.query_tokens = tokenize(query.query)
    query.num_query_tokens = len(query.query_tokens)
    # concept and stygrp counts
    query.concept_counts, query.stygroup_counts = qminer_annotate(query.query)
    return query


def generate_doc_features(doc: Doc, cache: shelve.DbfilenameShelf
                          ) -> Doc:
    if doc.is_populated:
        return doc
    # token counts
    doc.title_tokens = tokenize(doc.title)
    doc.num_title_tokens = len(doc.title_tokens)
    doc.section_title_tokens = tokenize(doc.section_title)
    doc.num_section_title_tokens = len(doc.section_title_tokens)
    doc.bread_crumb_tokens = tokenize(doc.bread_crumb.replace(" | ", " "))
    doc.num_bread_crumb_tokens = len(doc.bread_crumb_tokens)
    doc.text_tokens = tokenize(doc.text)
    doc.num_text_tokens = len(doc.text_tokens)
    # term statisitcs
    index_stats = compute_index_statistics(doc.doc_id, elastic)
    doc.title_ttf = index_stats["title"]["sum_ttf"]
    doc.title_min_tf = index_stats["title"]["min_tf"]
    doc.title_max_tf = index_stats["title"]["max_tf"]
    doc.title_mean_tf = index_stats["title"]["mean_tf"]
    doc.title_var_tf = index_stats["title"]["var_tf"]
    doc.title_min_tfidf = index_stats["title"]["min_tfidf"]
    doc.title_max_tfidf = index_stats["title"]["max_tfidf"]
    doc.title_mean_tfidf = index_stats["title"]["mean_tfidf"]
    doc.title_var_tfidf = index_stats["title"]["var_tfidf"]
    doc.section_title_ttf = index_stats["section_title"]["sum_ttf"]
    doc.section_title_min_tf = index_stats["section_title"]["min_tf"]
    doc.section_title_max_tf = index_stats["section_title"]["max_tf"]
    doc.section_title_mean_tf = index_stats["section_title"]["mean_tf"]
    doc.section_title_var_tf = index_stats["section_title"]["var_tf"]
    doc.section_title_min_tfidf = index_stats["section_title"]["min_tfidf"]
    doc.section_title_max_tfidf = index_stats["section_title"]["max_tfidf"]
    doc.section_title_mean_tfidf = index_stats["section_title"]["mean_tfidf"]
    doc.section_title_var_tfidf = index_stats["section_title"]["var_tfidf"]
    doc.bread_crumb_ttf = index_stats["bread_crumb"]["sum_ttf"]
    doc.bread_crumb_min_tf = index_stats["bread_crumb"]["min_tf"]
    doc.bread_crumb_max_tf = index_stats["bread_crumb"]["max_tf"]
    doc.bread_crumb_mean_tf = index_stats["bread_crumb"]["mean_tf"]
    doc.bread_crumb_var_tf = index_stats["bread_crumb"]["var_tf"]
    doc.bread_crumb_min_tfidf = index_stats["bread_crumb"]["min_tfidf"]
    doc.bread_crumb_max_tfidf = index_stats["bread_crumb"]["max_tfidf"]
    doc.bread_crumb_mean_tfidf = index_stats["bread_crumb"]["mean_tfidf"]
    doc.bread_crumb_var_tfidf = index_stats["bread_crumb"]["var_tfidf"]
    doc.text_ttf = index_stats["text"]["sum_ttf"]
    doc.text_min_tf = index_stats["text"]["min_tf"]
    doc.text_max_tf = index_stats["text"]["max_tf"]
    doc.text_mean_tf = index_stats["text"]["mean_tf"]
    doc.text_var_tf = index_stats["text"]["var_tf"]
    doc.text_min_tfidf = index_stats["text"]["min_tfidf"]
    doc.text_max_tfidf = index_stats["text"]["max_tfidf"]
    doc.text_mean_tfidf = index_stats["text"]["mean_tfidf"]
    doc.text_var_tfidf = index_stats["text"]["var_tfidf"]
    # set to populated
    doc.is_populated = True
    return doc


def generate_query_doc_features(query: Query,
                                doc: Doc,
                                elastic: Elasticsearch,
                                sts_encoder: SentenceTransformer
                                ) -> Doc:
    field_names = ["title", "section_title", "bread_crumb", "text"]
    # token counts overlap
    query_tokens = set(query.query_tokens)
    doc.title_token_overlap = len(query_tokens.intersection(
        set(doc.title_tokens)))
    doc.section_title_token_overlap = len(query_tokens.intersection(
        set(doc.section_title_tokens)))
    doc.bread_crumb_token_overlap = len(query_tokens.intersection(
        set(doc.bread_crumb_tokens)))
    doc.text_token_overlap = len(query_tokens.intersection(set(doc.text_tokens)))
    # concept and stygroup overlap
    doc.title_concept_counts, doc.title_stygroup_counts = qminer_annotate(
        doc.title, filter_concepts=query.concept_counts.keys())
    doc.section_title_concept_counts, doc.section_title_stygroup_counts = \
        qminer_annotate(doc.section_title, 
                        filter_concepts=query.concept_counts.keys())
    doc.bread_crumb_concept_counts, doc.bread_crumb_stygroup_counts = \
        qminer_annotate(doc.bread_crumb, 
                        filter_concepts=query.concept_counts.keys())
    doc.text_concept_counts, doc.text_stygroup_counts = qminer_annotate(
        doc.text, filter_concepts=query.concept_counts.keys())
    # concept overlap
    doc.title_concept_overlap = compute_weighted_overlap(
        query.concept_counts, doc.title_concept_counts)
    doc.section_title_concept_overlap = compute_weighted_overlap(
        query.concept_counts, doc.section_title_concept_counts)
    doc.bread_crumb_concept_overlap = compute_weighted_overlap(
        query.concept_counts, doc.bread_crumb_concept_counts)
    doc.text_concept_overlap = compute_weighted_overlap(
        query.concept_counts, doc.text_concept_counts)    
    ## stygrp overlap
    doc.title_stygroup_overlap = compute_weighted_overlap(
        query.stygroup_counts, doc.title_stygroup_counts)
    doc.section_title_stygroup_overlap = compute_weighted_overlap(
        query.stygroup_counts, doc.section_title_stygroup_counts)
    doc.bread_crumb_stygroup_overlap = compute_weighted_overlap(
        query.stygroup_counts, doc.bread_crumb_stygroup_counts)
    doc.text_stygroup_overlap = compute_weighted_overlap(
        query.stygroup_counts, doc.text_stygroup_counts)
    # BM25 scores
    bm25_scores = compute_bm25_score(query.query, doc, 
                                     field_names,
                                     elastic)
    doc.title_bm25_score = bm25_scores["title"]
    doc.section_title_bm25_score = bm25_scores["section_title"]
    doc.bread_crumb_bm25_score = bm25_scores["bread_crumb"]
    doc.text_bm25_score = bm25_scores["text"]
    # vector cosine similarity scores
    vector_sims = compute_vector_similarity(
        query.query, doc, sts_encoder)
    doc.title_vec_score = vector_sims["title"]
    doc.section_title_vec_score = vector_sims["section_title"]
    doc.bread_crumb_vec_score = vector_sims["bread_crumb"]
    doc.text_vec_score = vector_sims["text"]
    return doc


def generate_features(query: str, doc_id: str,
                      cache: shelve.DbfilenameShelf,
                      elastic: Elasticsearch,
                      sts_encoder: SentenceTransformer,
                      judgement: str,
                      feats_dict: Dict[str, Dict[str, Any]]
                      ) -> Tuple[Query, Doc]:

    key = "|".join([query, doc_id])
    if key in feats_dict.keys():
        return feats_dict[key]

    query = Query(query=query)
    query = generate_query_features(query)

    doc = get_doc(doc_id, elastic, cache)
    doc = generate_doc_features(doc, cache)
    cache[doc.doc_id] = doc

    doc = generate_query_doc_features(query, doc, elastic, sts_encoder)

    # generate features dictioanry
    feats = {
        "num_query_tokens": query.num_query_tokens,

        "num_title_tokens": doc.num_title_tokens,
        "num_section_title_tokens": doc.num_section_title_tokens,
        "num_bread_crumb_tokens": doc.num_bread_crumb_tokens,
        "num_text_tokens": doc.num_text_tokens,
        
        "num_title_token_overlap": doc.title_token_overlap,
        "num_section_title_token_overlap": doc.section_title_token_overlap,
        "num_bread_crumb_token_overlap": doc.bread_crumb_token_overlap,
        "num_text_token_overlap": doc.text_token_overlap,
        
        "title_concept_overlap": doc.title_concept_overlap,
        "section_title_concept_overlap": doc.section_title_concept_overlap,
        "bread_crumb_concept_overlap": doc.bread_crumb_concept_overlap,
        "text_concept_overlap": doc.text_concept_overlap,
        
        "title_stygroup_overlap": doc.title_stygroup_overlap,
        "section_title_stygroup_overlap": doc.section_title_stygroup_overlap,
        "bread_crumb_stygroup_overlap": doc.bread_crumb_stygroup_overlap,
        "text_stygroup_overlap": doc.text_stygroup_overlap,

        "title_bm25_score": doc.title_bm25_score,
        "section_title_bm25_score": doc.section_title_bm25_score,
        "bread_crumb_bm25_score": doc.bread_crumb_bm25_score,
        "text_bm25_score": doc.text_bm25_score,

        "title_ttf": doc.title_ttf,
        "title_min_tf": doc.title_min_tf,
        "title_max_tf": doc.title_max_tf,
        "title mean_tf": doc.title_mean_tf,
        "title_var_tf": doc.title_var_tf,

        "title_min_tfidf": doc.title_min_tfidf,
        "title_max_tfidf": doc.title_max_tfidf,
        "title mean_tfidf": doc.title_mean_tfidf,
        "title_var_tfidf": doc.title_var_tfidf,

        "section_title_ttf": doc.section_title_ttf,
        "section_title_min_tf": doc.section_title_min_tf,
        "section_title_max_tf": doc.section_title_max_tf,
        "section_title mean_tf": doc.section_title_mean_tf,
        "section_title_var_tf": doc.section_title_var_tf,

        "section_title_min_tfidf": doc.section_title_min_tfidf,
        "section_title_max_tfidf": doc.section_title_max_tfidf,
        "section_title mean_tfidf": doc.section_title_mean_tfidf,
        "section_title_var_tfidf": doc.section_title_var_tfidf,

        "bread_crumb_ttf": doc.bread_crumb_ttf,
        "bread_crumb_min_tf": doc.bread_crumb_min_tf,
        "bread_crumb_max_tf": doc.bread_crumb_max_tf,
        "bread_crumb mean_tf": doc.bread_crumb_mean_tf,
        "bread_crumb_var_tf": doc.bread_crumb_var_tf,

        "bread_crumb_min_tfidf": doc.bread_crumb_min_tfidf,
        "bread_crumb_max_tfidf": doc.bread_crumb_max_tfidf,
        "bread_crumb mean_tfidf": doc.bread_crumb_mean_tfidf,
        "bread_crumb_var_tfidf": doc.bread_crumb_var_tfidf,

        "text_ttf": doc.text_ttf,
        "text_min_tf": doc.text_min_tf,
        "text_max_tf": doc.text_max_tf,
        "text_mean_tf": doc.text_mean_tf,
        "text_var_tf": doc.text_var_tf,

        "text_min_tfidf": doc.text_min_tfidf,
        "text_max_tfidf": doc.text_max_tfidf,
        "text mean_tfidf": doc.text_mean_tfidf,
        "text_var_tfidf": doc.text_var_tfidf,

        "title_vec_score": doc.title_vec_score,
        "section_title_vec_score": doc.section_title_vec_score,
        "bread_crumb_vec_score": doc.bread_crumb_vec_score,
        "text_vec_score": doc.text_vec_score,
    }
    if judgement is not None:
        feats["label"] = 1 if judgement == "RELEVANT" else 0

    return feats


def save_feature(query: str, doc_id: str, feats: Dict[str, Any],
                 feats_dict: Dict[str, Dict[str, Any]]
                 ) -> None:
    key = "|".join([query, doc_id])
    if key not in feats_dict.keys():
        feats_dict[key] = feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments_file", type=str, required=True,
                        help="Judgments file")
    parser.add_argument("--features_file", type=str, required=True,
                        help="Features file")
    parser.add_argument("--judgment_type", type=str, required=True,
                        choices=["pointwise", "pairwise", "scored"],
                        help="Judgment type")
    args = parser.parse_args()
    
    judgments_file = args.judgments_file
    features_file = args.features_file
    judgement_type = args.judgment_type

    _ = load_dotenv(find_dotenv())

    es_url = os.environ["ES_URL"]
    es_index = os.environ["ES_INDEX"]
    qminer_url = os.environ["QMINER_URL"]

    elastic = Elasticsearch([es_url])
    sts_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    num_read = 0
    feats_dict = {}
    with open(judgments_file, "r") as fin, \
         open(features_file, "w") as fout:
        for line in fin:
            if line.startswith("#QUESTION"):
                continue
            if num_read % 1000 == 0:
                print("read {:d} lines".format(num_read))

            if judgement_type == "pointwise" or judgement_type == "scored":
                query, doc_id, judgement = line.strip().split("\t")
                feats = generate_features(query, doc_id, elastic, sts_encoder,
                                          judgement, feats_dict)
                fout.write(json.dumps(feats, cls=NpEncoder) + "\n")
                save_feature(query, doc_id, feats, feats_dict)
            else:
                query, doc_1_id, doc_2_id, judgement = line.strip().split("\t")
                feats = generate_features(query, doc_1_id, elastic, sts_encoder,
                                          None, feats_dict)
                fout.write(json.dumps(feats, cls=NpEncoder) + "\n")
                save_feature(query, doc_1_id, feats, feats_dict)
                feats = generate_features(query, doc_2_id, elastic, sts_encoder,
                                          None, feats_dict)
                fout.write(json.dumps(feats, cls=NpEncoder) + "\n")
                save_feature(query, doc_2_id, feats, feats_dict)

            num_read += 1
