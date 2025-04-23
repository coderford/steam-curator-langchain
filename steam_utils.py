# References:
# https://partner.steamgames.com/doc/store/getreviews
# https://medium.com/codex/efficiently-scraping-steam-game-reviews-with-python-a-comprehensive-guide-3a5732cb7f0b
# https://stackoverflow.com/a/78263053

import argparse
import os
import json
import re
import requests

import glog as log


def get_game_id_from_url(game_url):
    """
    Extracts the app ID from a Steam game URL.

    Args:
        game_url: The URL of the game's store page.

    Returns:
        The game ID as an integer.
    """
    start = game_url.find("app/") + len("app/")
    end = game_url.find("/", start)
    return int(game_url[start:end])


def get_game_title_from_url(url, replace_underscore=True):
    """
    Extracts the game title from a Steam game URL.

    Args:
        url: The URL of the game's store page.

    Returns:
        The game title as a string.
    """
    match = re.search(r"/app/\d+/(.+?)/", url)
    if match:
        return match.group(1).replace("_", " ") if replace_underscore else match.group(1)
    else:
        log.warning("URL does not contain a valid game title")
        return ""


def get_user_reviews(app_id, language="english", num_per_page=20, filter="recent", review_type="all", purchase_type="all", limit=20):
    """
    Fetches user reviews for a given Steam app ID.

    Args:
        app_id: The ID of the Steam app.
        language: The language of the reviews to fetch (e.g., "english").
        num_per_page: The number of reviews to fetch per page.
        filter: The filter to apply to the reviews (e.g., "recent").
        review_type: The type of reviews to fetch (e.g., "all").
        purchase_type: The type of purchase to filter by (e.g., "all").
        limit: The maximum number of reviews to fetch.

    Returns:
        A list of dictionaries containing data about user reviews.
    """
    params = {
        "json": 1,
        "language": language,
        "cursor": "*",
        "num_per_page": num_per_page,
        "filter": filter,
        "review_type": review_type,
        "purchase_type": purchase_type,
    }
    user_review_url = f"https://store.steampowered.com/appreviews/{app_id}"
    user_reviews = []
    reviews_summary = {}

    while len(user_reviews) < limit:
        try:
            response = requests.get(user_review_url, params=params)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch user reviews. Status code: {response.status_code}")

            response_json = response.json()
            user_reviews.extend(response_json["reviews"])  # Add each review to the list

            if not reviews_summary:
                reviews_summary = {
                    "review_score_desc": response_json.get("query_summary", {}).get("review_score_desc", ""),
                    "total_positive": response_json.get("query_summary", {}).get("total_positive", 0),
                    "total_negative": response_json.get("query_summary", {}).get("total_negative", 0),
                    "total_reviews": response_json.get("query_summary", {}).get("total_reviews", 0),
                }

        except requests.exceptions.HTTPError as e:
            log.error(f"HTTP error occurred: {e}")
            return user_reviews
        except json.JSONDecodeError as e:
            log.error(f"JSON decoding error occurred: {e}")
            return user_reviews
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}")
            return user_reviews

        if len(response_json["reviews"]) > 0:
            log.info(f"Fetched {len(user_reviews)} reviews so far...")

        new_cursor = response_json.get("cursor", "")
        if new_cursor == params["cursor"]:
            log.info(f"Found no update in cursor after fetching {len(user_reviews)} reviews")
            break

        params["cursor"] = new_cursor
        if not params["cursor"]:
            log.info(f"Reached the end of all reviews after fetching {len(user_reviews)} reviews")
            break

        if len(response_json["reviews"]) == 0:
            log.info(f"Got 0 reviews after fetching {len(user_reviews)} reviews, stopping...")
            break
    
    # Keep only unique reviews from user_reviews
    seen_review_ids = set()
    unique_user_reviews = []
    for review in user_reviews:
        review_id = review["recommendationid"]
        if review_id not in seen_review_ids:
            seen_review_ids.add(review_id)
            unique_user_reviews.append(review)

    if len(unique_user_reviews) < len(user_reviews):
        log.info(f"Removed {len(user_reviews) - len(unique_user_reviews)} duplicate reviews.")
    return {
        "query_summary": reviews_summary,
        "reviews": unique_user_reviews,
    }


def get_game_details(app_id, cc="IN"):
    """
    Fetches the details of a Steam game based on its app ID. Retrieves information such as game title, description, release date, and genre.

    Args:
        app_id: The ID of the Steam app.
        cc: The country code for the store (e.g., "IN" for India).

    Returns:
        A dictionary containing the game details. Returns an empty dictionary if an error occurs.
    """
    game_details_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc={cc}"
    try:
        response = requests.get(game_details_url)
        if response.status_code != 200:
            raise Exception("Failed to fetch game details. Status code:", response.status_code)

        response_json = response.json()
        if response_json[str(app_id)]["success"]:
            game_data = response_json[str(app_id)]["data"]
            return game_data
        else:
            log.error("Unknown error occurred while trying fetch game details.")
            return {}
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP error occurred: {e}")
    except json.JSONDecodeError as e:
        log.error(f"JSON decoding error occurred: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred while fetching game details: {e}")
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Steam game details and user reviews.')
    parser.add_argument('--game_url', type=str, default="https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/", help='Steam store url for game')
    args = parser.parse_args()

    game_id = get_game_id_from_url(args.game_url)
    game_title = get_game_title_from_url(args.game_url, replace_underscore=False)
    game_details = get_game_details(game_id)
    reviews_data = get_user_reviews(game_id, language="english", num_per_page=50, limit=500)

    with open(f"steam_reviews_{game_id}_{game_title}.json", "w") as f:
        json.dump(reviews_data, f, indent=4)
    with open(f"steam_details_{game_id}_{game_title}.json", "w") as f:
        json.dump(game_details, f, indent=4)
