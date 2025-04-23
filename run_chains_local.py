import argparse
import json

import glog as log
from langchain.chains.sequential import SequentialChain
from langchain.globals import set_verbose, set_debug
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
    ).get("reviews", [])

    deterministic_filter = filter_chains.DeterministicFilterChain()

    llm = ChatOllama(model=args.model, temperature=0.7)
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

    log.info(f"Kept {len(filtered_reviews)} out of {len(reviews)} reviews")

    original_output_file = f"original_reviews_{args.app_id}.json"
    filtered_output_file = f"filtered_reviews_{args.app_id}.json"

    log.info(f"Saving filtered review data to {filtered_output_file}")
    with open(filtered_output_file, "w") as f:
        json.dump(filtered_reviews, f, indent=4)
    
    log.info(f"Saving original reviews to {original_output_file}")
    with open(original_output_file, "w") as f:
        json.dump(reviews, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter reviews")
    parser.add_argument("app_id", type=str, help="Steam app ID")
    parser.add_argument("--model", type=str, default="gemma3:4b")
    parser.add_argument("--num_reviews", type=int, default=20, help="Number of reviews to filter")
    parser.add_argument("--language", type=str, default="english", help="Language for reviews")
    parser.add_argument("--num_per_page", type=int, default=20, help="Number of reviews per page")
    parser.add_argument("--filter", type=str, default="recent", help="Filter for reviews. Can be 'all' or 'recent'.")
    parser.add_argument(
        "--review_type", type=str, default="all", help="Review type. Can be 'positive', 'negative' or 'all'."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_debug(args.debug)
    main(args)
