import argparse
import json

import glog as log
from langchain.chains.sequential import SequentialChain
from langchain.globals import set_verbose, set_debug, set_llm_cache
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.cache import SQLiteCache
from langchain_core.runnables import RunnableLambda, RunnableParallel

import output_parsers
import steam_utils
from chains import filter_chains, summarization_chains, aggregation_chains
from prompts import filter_prompts, summarization_prompts, aggregation_prompts


def get_filter_chain(model, temperature=0.7):
    deterministic_filter = filter_chains.DeterministicFilterChain()

    filter_llm = ChatOllama(model=model, temperature=temperature)
    llm_filter = filter_chains.LLMFilterChain(
        filter_llm,
        output_parser=output_parsers.FILTER_CHAIN_PARSER,
        prompt_template=filter_prompts.FLUFF_FILTER_PROMPT,
    )

    remap_output = RunnableLambda(lambda x: {"remapped_reviews": x["filtered_reviews"]})
    remap_input = RunnableLambda(lambda x: {"reviews": x["remapped_reviews"]})

    filter_chain = deterministic_filter | remap_output | remap_input | llm_filter
    return filter_chain


def get_summarization_chain(model, temperature=0.7):
    summary_llm = ChatOllama(model=model, temperature=temperature)
    summarization_chain = summarization_chains.SummarizationChain(
        summary_llm,
        output_parser=output_parsers.JUICE_SUMMARIZATION_CHAIN_PARSER,
        prompt_template=summarization_prompts.JUICE_SUMMARIZATION_PROMPT,
    )
    return summarization_chain


def get_aggregation_chain(model, temperature=0.7):
    aggregation_llm = ChatOllama(model=model, temperature=temperature)
    aspects = list(aggregation_prompts.JUICE_AGGREGATION_PROMPTS.keys())
    aggregation_branches = {}
    for aspect in aspects:
        remap_input = RunnableLambda(
            (lambda y: (lambda x: {"batch_summaries": x["batch_summaries"], "summary_aspect": y}))(aspect)
        )
        chain = aggregation_chains.AggregationChain(
            aggregation_llm,
            output_parser=output_parsers.JUICE_AGGREGATION_CHAIN_PARSER,
            prompt_template=aggregation_prompts.JUICE_AGGREGATION_PROMPTS[aspect],
        )
        aggregation_branches[aspect] = remap_input | chain

    aggregation_chain = RunnableParallel(branches=aggregation_branches)
    return aggregation_chain


def main(args):
    reviews = steam_utils.get_user_reviews(
        args.app_id,
        limit=args.num_reviews,
        language=args.language,
        num_per_page=args.num_per_page,
        filter=args.filter,
        review_type=args.review_type,
    ).get("reviews", [])

    filter_chain = get_filter_chain(args.filter_model)
    summarization_chain = get_summarization_chain(args.summarization_model)
    aggregation_chain = get_aggregation_chain(args.aggregation_model)
    complete_chain = filter_chain | summarization_chain | aggregation_chain

    chain_output = complete_chain.invoke({"reviews": reviews})
    output_file = f"chain_output_{args.app_id}.json"

    log.info(f"Saving chain output data to {output_file}")
    with open(output_file, "w") as f:
        json.dump(chain_output, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter reviews")
    parser.add_argument("app_id", type=str, help="Steam app ID")
    parser.add_argument("--filter_model", type=str, default="gemma3:4b")
    parser.add_argument("--summarization_model", type=str, default="qwen2.5:7b")
    parser.add_argument("--aggregation_model", type=str, default="qwen2.5:7b")
    parser.add_argument("--num_reviews", type=int, default=20, help="Number of reviews to filter")
    parser.add_argument("--language", type=str, default="english", help="Language for reviews")
    parser.add_argument("--num_per_page", type=int, default=20, help="Number of reviews per page")
    parser.add_argument("--filter", type=str, default="recent", help="Filter for reviews. Can be 'all' or 'recent'.")
    parser.add_argument(
        "--review_type", type=str, default="all", help="Review type. Can be 'positive', 'negative' or 'all'."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--skip_cache", action="store_true", help="Skip caching local db")
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_debug(args.debug)
    if not args.skip_cache:
        set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
    main(args)
