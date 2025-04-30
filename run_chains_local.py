import argparse
import json
import os
from datetime import datetime

import pandas as pd
import glog as log
from langchain.chains.sequential import SequentialChain
from langchain.globals import set_verbose, set_debug, set_llm_cache
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.cache import SQLiteCache
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.exceptions import OutputParserException
from tqdm import tqdm

import output_parsers
import steam_utils
from chains import filter_chains, summarization_chains, aggregation_chains
from prompts import filter_prompts, summarization_prompts, aggregation_prompts


class OverwriteSQLiteCache(SQLiteCache):
    def lookup(self, prompt: str, llm_string):
        # Always return None to force recompute
        return None


def get_blurb(review_text, model="qwen2.5:7b"):
    """
    Generates a blurb for a given review text using the specified language model.

    Args:
        review_text (str): The review text to generate a blurb for.
        model (str, optional): The language model to use for generating the blurb. Defaults to "qwen2.5:7b".

    Returns:
        str: The generated blurb.
    """
    blurb_prompt = ChatPromptTemplate.from_template(aggregation_prompts.BLURB_PROMPT)
    llm = ChatOllama(model=model, temperature=0.0)

    chain = blurb_prompt | llm | StrOutputParser()
    output = chain.invoke({"review_text": review_text})
    return output


def get_filter_chain(model, temperature=0.0):
    """
    Creates a filter chain to determine whether reviews should be included based on certain criteria.

    Args:
        model (str): The language model to use for filtering.
        temperature (float, optional): The temperature parameter for the language model. Defaults to 0.0.

    Returns:
        Chain: A LangChain chain that includes deterministic filtering and LLM-based filtering.
    """
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


def get_summarization_chain(model, temperature=0.0, batch_size=12):
    """
    Creates a summarization chain to generate summaries of the filtered reviews.

    Args:
        model (str): The language model to use for summarization.
        temperature (float, optional): The temperature parameter for the language model. Defaults to 0.0.

    Returns:
        Chain: A LangChain chain that performs LLM-based summarization of reviews.
    """
    summary_llm = ChatOllama(model=model, temperature=temperature)
    summarization_chain = summarization_chains.SummarizationChain(
        summary_llm,
        output_parser=output_parsers.JUICE_SUMMARIZATION_CHAIN_PARSER,
        prompt_template=summarization_prompts.JUICE_SUMMARIZATION_PROMPT,
        batch_size=batch_size,
    )
    return summarization_chain


def get_aggregation_chain(model, temperature=0.0):
    """
    Creates an aggregation chain to generate summaries of the filtered reviews for each aspect.

    Args:
        model (str): The language model to use for aggregation.
        temperature (float, optional): The temperature parameter for the language model. Defaults to 0.0.

    Returns:
        Chain: A LangChain chain that performs LLM-based aggregation of review summaries and output parsing.
    """
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
        ).with_retry(stop_after_attempt=3, retry_if_exception_type=[OutputParserException])
        aggregation_branches[aspect] = remap_input | chain

    aggregation_chain = RunnableParallel(branches=aggregation_branches)
    return aggregation_chain


def make_complete_chain(filter_model="gemma3:4b", summarization_model="qwen2.5:7b", aggregation_model="gemma3:12b", summarization_batch_size=12):
    """
    Creates a complete chain that filters, summarizes, and aggregates reviews using specified language models.

    Args:
        filter_model (str): The language model to use for filtering reviews.
        summarization_model (str): The language model to use for summarizing reviews.
        aggregation_model (str): The language model to use for aggregating review summaries.

    Returns:
        Chain: A LangChain chain that filters, summarizes, and aggregates reviews based on the specified models.
    """
    filter_chain = get_filter_chain(filter_model)
    summarization_chain = get_summarization_chain(summarization_model, batch_size=summarization_batch_size)
    aggregation_chain = get_aggregation_chain(aggregation_model)
    complete_chain = filter_chain | summarization_chain | aggregation_chain
    return complete_chain


def run_for_app_id(
    app_id,
    complete_chain,
    num_reviews=200,
    num_per_page=100,
    language="english",
    review_filter="recent",
    review_type="all",
    allow_other_languages=True,
):
    reviews = steam_utils.get_user_reviews(
        app_id,
        limit=num_reviews,
        num_per_page=num_per_page,
        language=language,
        filter=review_filter,
        review_type=review_type,
    ).get("reviews", [])

    # if number of reviews is < num_revies, try with languages = 'all'
    if len(reviews) < num_reviews and allow_other_languages:
        reviews = steam_utils.get_user_reviews(
            app_id,
            limit=num_reviews,
            num_per_page=num_per_page,
            language="all",
            filter=review_filter,
            review_type=review_type,
        ).get("reviews", [])

    try:
        chain_output = complete_chain.invoke({"reviews": reviews})
    except OutputParserException as e:
        raw = getattr(e, "llm_output", e.args[1] if len(e.args) > 1 else None)
        print("❗️ Failed to parse JSON. Raw LLM output was:\n", raw)
        raise

    branches = chain_output["branches"]
    total_score = 0
    score_breakdown_text = ""
    for aspect in branches:
        aspect_capitalized = aspect.replace("_", " ").capitalize()
        aspect_score = branches[aspect]["aggregate_score"]
        total_score += aspect_score
        score_breakdown_text += f"{aspect_capitalized} ({aspect_score}/10): {branches[aspect]['score_explanation']}\n\n"
    final_score = total_score / len(branches.keys())

    blurb = get_blurb(score_breakdown_text, model=args.blurb_model)
    blurb = f"JUICE Score: {final_score:.1f}. {blurb}"
    chain_output["final_score"] = final_score
    chain_output["score_breakdown_text"] = score_breakdown_text
    chain_output["blurb"] = blurb
    return chain_output


def main(args):
    complete_chain = make_complete_chain(
        filter_model=args.filter_model,
        summarization_model=args.summarization_model,
        aggregation_model=args.aggregation_model,
        summarization_batch_size=args.summarization_batch_size,
    )

    if args.app_id:
        game_details = steam_utils.get_game_details(args.app_id)
        name_clean = "".join(e for e in game_details["name"] if e.isalnum() or e.isspace())
        name_clean = name_clean.replace(" ", "_")
        output_file = f"chain_outputs/chain_output_{args.app_id}_{name_clean}.json"
        os.makedirs("chain_outputs/", exist_ok=True)

        chain_output = run_for_app_id(
            args.app_id,
            complete_chain,
            num_reviews=args.num_reviews,
            num_per_page=args.num_per_page,
            language=args.language,
            review_filter=args.filter,
            review_type=args.review_type,
        )

        log.info(f"Saving chain output data to {output_file}")
        with open(output_file, "w") as f:
            json.dump(chain_output, f, indent=4)

        print(f"\nJUICE SCORE: {chain_output['final_score']:.1f}/10\n")
        print(chain_output["score_breakdown_text"])
        print("Blurb Version")
        print("-------------")
        print(chain_output["blurb"])
    else:
        app_ids = [x.strip() for x in open(args.run_for_file, "r").readlines()]
        tuples = []
        columns = [
            "app_id",
            "name",
            "url",
            "metacritic_score",
            "genres",
            "juice_score",
            "blurb",
        ]
        review_aspects = sorted(list(aggregation_prompts.JUICE_AGGREGATION_PROMPTS.keys()))
        for aspect in review_aspects:
            columns.append(f"{aspect}_score")
            columns.append(f"{aspect}_explanation")

        for app_id in tqdm(app_ids):
            try:
                game_details = steam_utils.get_game_details(app_id.strip())
            except Exception as e:
                log.exception(f"Error getting game details for app_id={args.app_id}: {e}")
                log.info(f"Skipping {app_id} due to error")
                continue
            genres = [g["description"] for g in game_details.get("genres", [])]
            log.info(f"Running for app_id {app_id}: {game_details['name']}")
            try:
                chain_output = run_for_app_id(
                    app_id,
                    complete_chain,
                    num_reviews=args.num_reviews,
                    num_per_page=args.num_per_page,
                    language=args.language,
                    review_filter=args.filter,
                    review_type=args.review_type,
                )
            except Exception as e:
                log.exception(f"Error running chain for app_id={args.app_id}: {e}")
                log.info(f"Skipping {app_id} due to error")
                continue
            datapoint = [
                app_id,
                game_details["name"],
                f"store.steampowered.com/app/{app_id}",
                game_details.get("metacritic", {}).get("score", None),
                json.dumps(genres),
                chain_output["final_score"],
                chain_output["blurb"],
            ]
            for aspect in review_aspects:
                aspect_score = chain_output["branches"][aspect]["aggregate_score"]
                aspect_explanation = chain_output["branches"][aspect]["score_explanation"]
                datapoint.extend([aspect_score, aspect_explanation])
            tuples.append(datapoint)

        df = pd.DataFrame(tuples, columns=columns)
        output_file = f"run_results_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.csv"
        log.info(f"Saving results to {output_file}")
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter reviews")
    me_group = parser.add_mutually_exclusive_group(required=True)
    me_group.add_argument("--app_id", type=str, help="Steam app ID")
    me_group.add_argument("--run_for_file", type=str, help="Path to file containing list of app IDs")
    parser.add_argument("--filter_model", type=str, default="qwen2.5-coder:7b")
    parser.add_argument("--summarization_model", type=str, default="granite3.3:8b")
    parser.add_argument("--summarization_batch_size", type=int, default=12)
    parser.add_argument("--aggregation_model", type=str, default="gemma3:12b")
    parser.add_argument("--blurb_model", type=str, default="qwen2.5:7b")
    parser.add_argument("--num_reviews", type=int, default=200, help="Number of reviews to filter")
    parser.add_argument("--language", type=str, default="english", help="Language for reviews")
    parser.add_argument("--num_per_page", type=int, default=100, help="Number of reviews per page")
    parser.add_argument("--filter", type=str, default="recent", help="Filter for reviews. Can be 'all' or 'recent'.")
    parser.add_argument(
        "--review_type", type=str, default="all", help="Review type. Can be 'positive', 'negative' or 'all'."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--skip_cache", action="store_true", help="Skip caching local db")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite cache instead of using it for lookups"
    )
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_debug(args.debug)
    if not args.skip_cache:
        if not args.overwrite_cache:
            set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
        else:
            set_llm_cache(OverwriteSQLiteCache(database_path=".langchain_cache.db"))
    main(args)
