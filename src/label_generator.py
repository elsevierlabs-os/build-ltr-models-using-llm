import argparse
import json
import os
import re
import requests

from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any

DATA_DIR = "../data"
HEADERS = {"Content-type": "application/json"}
TOP_K = 10


def get_topk_results(query: str,
                     doc_retrieval_url: str,
                     top_k: int = TOP_K) -> List[Dict[str, Any]]:
    post_body = {
        "query": query,
        "top_k": top_k,
    }
    response = requests.post(doc_retrieval_url, headers=HEADERS,
                             data=json.dumps(post_body))
    return response.json()["context"]


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


def get_relevance_pair_judgment(question: str,
                                lhs_text: str,
                                rhs_text: str,
                                prompt_template: str,
                                claude: Anthropic) -> str:
    prompt = prompt_template.format(query=question,
                                    document_1=lhs_text,
                                    document_2=rhs_text)
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
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--input-queries", type=str, required=True,
                          help="Input queries file")
    argparse.add_argument("--prompt-file", type=str, required=True,
                          help="Prompt template file")
    argparse.add_argument("--output-judgments", type=str, required=True,
                          help="Output Judgement file")
    argparse.add_argument("--judgment-type", type=str, required=True,
                          choices=["pointwise", "pairwise", "scored"],
                          help="Judgment type")
    argparse.add_argument("--top-k", type=int, default=10,
                          help="Top k results to retrieve")
    args = argparse.parse_args()

    query_file = args.input_queries
    prompt_file = args.prompt_file
    judgments_file = args.output_judgments
    judgment_type = args.judgment_type
    top_k = args.top_k

    _ = load_dotenv(find_dotenv())

    doc_retrieval_url = os.environ["DOC_RETRIEVAL_URL"]
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

    claude = Anthropic(api_key=anthropic_api_key)
    prompt_template = setup_prompt_template(prompt_file)

    with open(query_file, "r") as fin, \
         open(judgments_file, "w") as fout:
        if judgment_type == "pointwise" or judgment_type == "scored":
            fout.write("\t".join(["#QUERY", "DOC-ID", "JUDGMENT"]) + "\n")
        else:
            fout.write("\t".join(["#QUERY", "DOC-1-ID", "DOC-2-ID", "JUDGMENT"])
                       + "\n")
        for line in fin:
            query = line.strip()
            print("processing query: {:s}...", query)
            if query.startswith("#"):
                continue
            results = get_topk_results(query, doc_retrieval_url)
            if judgment_type == "pointwise" or judgment_type == "scored":
                for result in results:
                    doc_id = result["doc_id"]
                    text = result["text"]
                    judgment = get_relevance_judgement(
                        query, text, prompt_template)
                    fout.write("\t".join([query, doc_id, judgment]) + "\n")
            else:
                lhs_id = results[0]["doc_id"]
                lhs_text = results[0]["text"]
                rhs_texts = [(result["doc_id"], result["text"])
                             for result in results[1:]]
                for rhs_id, rhs_text in rhs_texts:
                    judgment = get_relevance_pair_judgment(
                        query, lhs_text, rhs_text, prompt_template)
                    fout.write("\t".join([query, lhs_id, rhs_id, judgment])
                               + "\n")
