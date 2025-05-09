from typing import List, Dict, Any

from langchain.chains.base import Chain
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from prompts import filter_prompts


class DeterministicFilterChain(Chain):
    min_words: int
    min_playtime: int
    ret_all_if_empty: bool

    @property
    def input_keys(self) -> List[str]:
        return ["reviews"]

    @property
    def output_keys(self) -> List[str]:
        return ["filtered_reviews"]

    def __init__(self, min_words: int = 5, min_playtime: int = 5 * 60, ret_all_if_empty: bool = True):
        super().__init__(min_words=min_words, min_playtime=min_playtime, ret_all_if_empty=ret_all_if_empty)
        self.min_words = min_words
        self.min_playtime = min_playtime
        self.ret_all_if_empty = ret_all_if_empty

    def is_review_too_small(self, review_text: str) -> bool:
        return len(review_text.split(" ")) < self.min_words

    def is_playtime_too_low(self, review_data: Dict[str, Any]) -> bool:
        # less than 10 minutes is too low for a juicy game
        playtime = review_data.get("author", {}).get("playtime_at_review", 36000)
        return playtime < self.min_playtime

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        filtered_reviews = []
        for review_data in inputs["reviews"]:
            if not self.is_review_too_small(review_data["review"]) and not self.is_playtime_too_low(review_data):
                filtered_reviews.append(review_data)
        if len(filtered_reviews) < 10 and self.ret_all_if_empty:
            return {"filtered_reviews": inputs["reviews"]}
        return {"filtered_reviews": filtered_reviews}


class LLMFilterChain(Chain):
    llm: BaseLanguageModel
    prompt_template: str
    output_parser: StructuredOutputParser
    enable_thinking: bool

    @property
    def input_keys(self) -> List[str]:
        return ["reviews"]

    @property
    def output_keys(self) -> List[str]:
        return ["filtered_reviews"]

    def __init__(
        self,
        llm: BaseLanguageModel,
        output_parser: StructuredOutputParser,
        prompt_template: str = filter_prompts.FLUFF_FILTER_PROMPT,
        enable_thinking: bool = False,
    ):
        super().__init__(
            llm=llm,
            output_parser=output_parser,
            prompt_template=prompt_template,
            enable_thinking=enable_thinking,
        )
        self.llm = llm
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        self.enable_thinking = enable_thinking

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        reviews = inputs["reviews"]
        prompt = ChatPromptTemplate([
            ("system", "" if self.enable_thinking else "/no_think"),
            ("human", self.prompt_template),
        ])
        format_instructions = self.output_parser.get_format_instructions()
        filter_chain = prompt | self.llm | self.output_parser

        filtered_reviews = []
        batch_inputs = []

        for review_data in reviews:
            batch_inputs.append(
                {"review_text": review_data["review"], "format_instructions": format_instructions}
            )

        outputs = filter_chain.batch(batch_inputs)
        assert len(outputs) == len(reviews)
        for review_data, output in zip(reviews, outputs):
            if " " in output["clean_review_text"]:
                review_data["review"] = output["clean_review_text"]
                filtered_reviews.append(review_data)

        return {"filtered_reviews": filtered_reviews}
