import argparse
import json

import glog as log
from langchain.chains.sequential import SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

import output_parsers
import steam_utils
from chains import filter_chains
from prompts import filter_prompts


def main(args):
    reviews = steam_utils.get_user_reviews(
        args.app_id,
        limit=args.num_reviews,
        language=args.language,
        num_per_page=args.num_per_page,
        filter=args.filter,
        review_type=args.review_type,
    )

    deterministic_filter = filter_chains.DeterministicFilterChain()

    llm = ChatOllama(model=args.model, temperature=0.7, verbose=args.verbose)
    llm_filter = filter_chains.LLMFilterChain(
        llm,
        output_parser=output_parsers.FILTER_CHAIN_PARSER,
        prompt_template=filter_prompts.FLUFF_FILTER_PROMPT,
    )

    remap_output = RunnableLambda(lambda x: {"remapped_reviews": x["filtered_reviews"]})
    remap_input = RunnableLambda(lambda x: {"reviews": x["remapped_reviews"]})
    complete_filter_chain = deterministic_filter | remap_output | remap_input | llm_filter


    log.info("Filtering reviews...")
    filter_data = complete_filter_chain.invoke({"reviews": reviews})
    filtered_reviews = filter_data["filtered_reviews"]

    if not args.output_file:
        args.output_file = "filtered_reviews.json"

    log.info(f"Saving filtered review data to {args.output_file}")
    with open(args.output_file, "w") as f:
        json.dump(filtered_reviews, f, indent=4)
    
    log.info("Saving original reviews to original_reviews.json")
    with open("original_reviews.json", "w") as f:
        json.dump(reviews, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter reviews")
    parser.add_argument("app_id", type=str, help="Steam app ID")
    parser.add_argument("--model", type=str, default="gemma3:4b")
    parser.add_argument("--num_reviews", type=int, default=20, help="Number of reviews to filter")
    parser.add_argument("--language", type=str, default="english", help="Language for reviews")
    parser.add_argument("--num_per_page", type=int, default=20, help="Number of reviews per page")
    parser.add_argument("--filter", type=str, default="all", help="Filter for reviews. Can be 'all' or 'recent'.")
    parser.add_argument(
        "--review_type", type=str, default="all", help="Review type. Can be 'positive', 'negative' or 'all'."
    )
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    args = parser.parse_args()
    main(args)
