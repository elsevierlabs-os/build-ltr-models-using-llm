import argparse
import numpy as np
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv
from elasticsearch import Elasticsearch
from langchain.prompts import PromptTemplate


def get_by_id_from_elastic(id: str,
                           elastic: Elasticsearch,
                           es_index: str) -> str:
    resp = elastic.search(index=es_index, query={
        "match": {
            "_id": id
        }
    })
    content = None
    for hit in resp["hits"]["hits"]:
        id = hit["_id"]
        content = hit["_source"]["content"]
        break
    return content


def flatten_text(text: str) -> str:
    if text is None:
        text = ""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_from_html_tag(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    matcher = re.search(pattern, text)
    if matcher is not None:
        return matcher.group(1)
    else:
        return None


def setup_prompt_template(prompt_file: str) -> PromptTemplate:
    with open(prompt_file, "r") as fin:
        template = fin.read()
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template


def parse_judgment(comp_str: str) -> str:
    matcher = re.search(r"<response>(.*?)</response>", comp_str)
    if matcher is not None:
        return matcher.group(1)
    else:
        return None


def get_relevance_judgement(question: str,
                            text: str,
                            prompt_template: str,
                            claude: Anthropic) -> str:
    prompt = prompt_template.format(query=question, document=text)
    completion = claude.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        stop_sequences=["</output>"],
        temperature=0.0,
        prompt=prompt
    )
    judgement = parse_judgment(completion.completion)
    return judgement


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-file", type=str, required=True,
                        help="Model Inference Output on Gold Set")
    parser.add_argument("--evaluation-file", type=str, required=True,
                        help="Evaluation Output Detail File")
    parser.add_argument("--prompt-file", type=str, required=True,
                        help="Prompt template file")
    args = parser.parse_args()

    inf_file = args.inference_file
    eval_file = args.evaluation_file
    prompt_file = args.prompt_file

    _ = load_dotenv(find_dotenv())

    es_host = os.environ.get("ES_HOST")
    es_index = os.environ.get("ES_INDEX")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    prompt_template = setup_prompt_template(prompt_file)
    elastic = Elasticsearch(es_host)
    claude = Anthropic(api_key=anthropic_api_key)

    prev_query = None
    avg_relevant, num_queries = 0, 0
    with open(inf_file, "r", encoding="urf-8") as fin, \
         open(eval_file, "w", encoding="utf-8") as fout:
        num_relevant = 0
        for line in fin:
            query, doc_id = line.strip().split("\t")
            if prev_query is not None and query != prev_query:
                print(prev_query, num_relevant)
                avg_relevant += num_relevant
                num_queries += 1
                num_relevant = 0

            text = get_by_id_from_elastic(doc_id, elastic, es_index)
            judgment = get_relevance_judgement(query, text, prompt_template,
                                               claude)
            if judgment == "relevant":
                num_relevant += 1
            prev_query = query

        print(prev_query, num_relevant)
        avg_relevant += num_relevant
        num_queries += 1

        print("Average relevant:", avg_relevant / num_queries)
