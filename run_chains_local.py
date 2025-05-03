import argparse
import json
import os
import time
from datetime import datetime

import pandas as pd
import glog as log
from langchain.globals import set_verbose, set_debug, set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.exceptions import OutputParserException
from tqdm import tqdm

import chain_utils
import steam_utils
from prompts import aggregation_prompts


class OverwriteSQLiteCache(SQLiteCache):
    def lookup(self, prompt: str, llm_string):
        # Always return None to force recompute
        return None


def _get_reviews(
    app_id,
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

    # if number of reviews is < num_revies, try to get rest with languages = 'all'
    if len(reviews) < num_reviews and allow_other_languages:
        log.info(f"Couldn't get enough reviews in {language}, trying to get the rest with all languages")
        remainder_reviews = steam_utils.get_user_reviews(
            app_id,
            limit=num_reviews - len(reviews),
            num_per_page=min(num_reviews - len(reviews), num_per_page),
            language="all",
            filter=review_filter,
            review_type=review_type,
        ).get("reviews", [])
        reviews.extend(remainder_reviews[:num_reviews - len(reviews)])
    return reviews


def calculate_weighted_juice_score(aspect_scores):
    aspect_weights = {
        "lore_worldbuilding_atmosphere": 0.25,
        "exploration": 0.25,
        "gameplay_mechanics": 0.25,
        "emotional_engagement": 0.1,
        "bloat_grinding": 0.15,
    }
    if set(aspect_scores.keys()) != set(aggregation_prompts.JUICE_AGGREGATION_PROMPTS.keys()):
        total_score = 0
        for aspect in aspect_score.keys():
            total_score += aspect_scores[aspect]
        return total_score / len(aspect_scores.keys())

    # bloat/grinding score should be capped to the max of the other scores
    max_area_scores = max([score for aspect, score in aspect_scores.items() if aspect != "bloat_grinding"])
    aspect_scores["bloat_grinding"] = min(aspect_scores["bloat_grinding"], max_area_scores)
    final_score = 0
    for aspect in aspect_scores.keys():
        final_score += aspect_weights[aspect] * aspect_scores[aspect]
    return final_score


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
    reviews = _get_reviews(
        app_id, num_reviews, num_per_page, language, review_filter, review_type, allow_other_languages
    )
    try:
        chain_output = complete_chain.invoke({"reviews": reviews})
    except OutputParserException as e:
        log.exception("❗️Failed to parse JSON output. Try re-running with debug mode")
        raise

    branches = chain_output["branches"]
    score_breakdown_text = ""
    aspect_scores = {}
    for aspect in branches:
        aspect_capitalized = aspect.replace("_", " ").capitalize()
        aspect_score = branches[aspect]["aggregate_score"]
        aspect_scores[aspect] = aspect_score
        score_breakdown_text += f"{aspect_capitalized} ({aspect_score}/10): {branches[aspect]['score_explanation']}\n\n"
    final_score = calculate_weighted_juice_score(aspect_scores)

    blurb = chain_utils.get_blurb(score_breakdown_text, model=args.blurb_model)
    blurb = f"JUICE Score: {final_score:.1f}. {blurb}"
    chain_output["final_score"] = final_score
    chain_output["score_breakdown_text"] = score_breakdown_text
    chain_output["blurb"] = blurb
    return chain_output


def main(args):
    complete_chain = chain_utils.make_complete_chain(
        filter_model=args.filter_model,
        summarization_model=args.summarization_model,
        aggregation_model=args.aggregation_model,
        summarization_batch_size=args.summarization_batch_size,
        club_reviews_batch_size=args.club_reviews_batch_size,
    )

    if args.app_id:
        game_details = steam_utils.get_game_details(args.app_id)
        name_clean = "".join(e for e in game_details["name"] if e.isalnum() or e.isspace())
        name_clean = name_clean.replace(" ", "_")
        output_file = f"chain_outputs/chain_output_{args.app_id}_{name_clean}.json"
        os.makedirs("chain_outputs/", exist_ok=True)

        start_time = time.time()
        chain_output = run_for_app_id(
            args.app_id,
            complete_chain,
            num_reviews=args.num_reviews,
            num_per_page=args.num_per_page,
            language=args.language,
            review_filter=args.filter,
            review_type=args.review_type,
        )
        log.info(f"Took {time.time()-start_time} seconds to run complete chain")

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
                log.exception(f"Error running chain for app_id={app_id}: {e}")
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
    parser.add_argument("--filter_model", type=str, default="qwen3:4b")
    parser.add_argument("--summarization_model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--summarization_batch_size", type=int, default=12, help="Batch size for summarization chain")
    parser.add_argument("--aggregation_model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--blurb_model", type=str, default="gemini-2.0-flash-lite")
    parser.add_argument("--num_reviews", type=int, default=300, help="Number of reviews to filter")
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
    parser.add_argument("--club_reviews_batch_size", type=int, default=4, help="Batch size for club reviews")
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_debug(args.debug)
    if not args.skip_cache:
        if not args.overwrite_cache:
            set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
        else:
            set_llm_cache(OverwriteSQLiteCache(database_path=".langchain_cache.db"))
    main(args)
