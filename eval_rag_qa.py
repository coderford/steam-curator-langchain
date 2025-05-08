import argparse
import json

from langchain.globals import set_verbose, set_debug
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain

import chain_utils
import rag_qa
import steam_utils


def generate_examples(docs):
    pass


def main(args):
    filter_chain = chain_utils.get_filter_chain("", club_reviews_batch_size=1)
    reviews = steam_utils.get_user_reviews(
        args.app_id, limit=args.num_reviews, num_per_page=min(args.num_reviews, 100)
    ).get("reviews", [])
    filtered_reviews = filter_chain.invoke({"reviews": reviews}).get("filtered_reviews", [])

    example_gen_chain = QAGenerateChain.from_llm(
        chain_utils.get_language_model(args.test_generation_model, temperature=args.temperature)
    )
    examples = example_gen_chain.batch(
        [{"doc": review_data["review"]} for review_data in filtered_reviews[: args.num_test_cases]]
    )
    examples = [x["qa_pairs"] for x in examples]

    rag_chain = rag_qa.make_retrieval_qa_chain(
        args.app_id,
        args.num_reviews,
        args.summarization_model,
        args.embedding_model,
        args.chat_model,
        temperature=args.temperature,
    )
    predictions = rag_chain.batch([{"input": x["query"]} for x in examples])

    eval_chain = QAEvalChain.from_llm(
        chain_utils.get_language_model(args.evaluation_model, temperature=args.temperature)
    )
    eval_results = eval_chain.evaluate(examples, predictions, prediction_key="answer")

    for i, eg in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + examples[i]['query'])
        print("Real Answer: " + examples[i]['answer'])
        print("Predicted Answer: " + predictions[i]['answer'])
        print("Predicted Grade: " + eval_results[i]['results'])
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("app_id", type=int, help="Steam app ID")
    parser.add_argument("--num_reviews", type=int, default=500)
    parser.add_argument("--summarization_model", type=str, default="gemini-2.0-flash-lite")
    parser.add_argument("--embedding_model", type=str, default="models/text-embedding-004")
    parser.add_argument("--chat_model", type=str, default="gemini-2.0-flash-lite")
    parser.add_argument("--test_generation_model", type=str, default="gemini-2.0-flash-lite")
    parser.add_argument("--evaluation_model", type=str, default="gemini-2.0-flash-lite")
    parser.add_argument("--num_test_cases", type=int, default=5, help="Number of test cases to generate")
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_debug(args.debug)

    main(args)
