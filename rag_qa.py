import argparse
import json
from typing import Iterator

import glog as log

from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.globals import set_verbose, set_debug, set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents.base import Document

import constants
import chain_utils
import steam_utils


def _disable_http_logging():
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class SteamReviewsLoader(BaseLoader):
    """Steam reviews loader."""

    app_id: int
    reviews: list[dict]
    num_reviews: int

    def __init__(self, app_id, num_reviews=100, num_per_page=100, summarization_model="qwen2.5:7b"):
        super().__init__()
        self.app_id = app_id

        raw_reviews = steam_utils.get_user_reviews(app_id, limit=num_reviews, num_per_page=num_per_page)["reviews"]
        self.reviews = (
            chain_utils.get_filter_chain("", club_reviews_batch_size=1)
            .invoke({"reviews": raw_reviews})
            .get("filtered_reviews")
        )
        self.num_reviews = len(self.reviews)

        # Also add aspect-wise summaries for better context
        filter_chain = chain_utils.get_filter_chain(model="", include_llm_filter=False, club_reviews_batch_size=3)
        summarization_chain = chain_utils.get_summarization_chain(model=summarization_model, temperature=0.7)
        summarization_output = (filter_chain | summarization_chain).invoke({"reviews": self.reviews})
        for batch_summary in summarization_output["batch_summaries"]:
            for key, value in batch_summary.items():
                self.reviews.append({"recommendationid": "", "review": f"{constants.ASPECT_NAMES[key]}: {value}"})

    def lazy_load(self) -> Iterator[Document]:
        for review in self.reviews:
            yield Document(page_content=review["review"], metadata={"recommendationid": review["recommendationid"]})


def main(args):
    loader = SteamReviewsLoader(args.app_id, num_reviews=args.num_reviews, summarization_model=args.summarization_model)
    embedder = chain_utils.get_embedding_model(args.embedding_model, temperature=0.7)

    db = DocArrayInMemorySearch.from_documents(loader.load(), embedder)
    retriever = db.as_retriever()

    llm = chain_utils.get_language_model(args.chat_model, temperature=0.7)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    if args.interactive:
        print("Type 'exit' or 'quit' to exit")

    if args.query:
        print(f"Q: {args.query}")
        answer = rag_chain.invoke({"input": args.query}).get("answer", "No answer found!")
        print(f"A: {answer}")

    if args.interactive:
        while True:
            query = input("Q: ")
            if query.lower() in ["exit", "quit"]:
                break
            answer = rag_chain.invoke({"input": query}).get("answer", "No answer found!")
            print(f"A: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("app_id", type=int, help="Steam App ID")
    parser.add_argument("--summarization_model", type=str, default="gemini-2.0-flash-lite")
    parser.add_argument("--embedding_model", type=str, default="models/text-embedding-004")
    parser.add_argument("--chat_model", type=str, default="gemini-2.0-flash-lite")
    parser.add_argument("--num_reviews", type=int, default=500, help="Number of reviews to retrieve")
    parser.add_argument("-q", "--query", default="", help="Query to search for")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive loop mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_debug(args.debug)
    set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

    _disable_http_logging()
    main(args)
