from typing import List, Dict, Any

from langchain.chains.base import Chain
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from prompts import summarization_prompts


class SummarizationChain(Chain):
    llm: BaseLanguageModel
    prompt_template: str
    output_parser: StructuredOutputParser
    batch_size: int
    enable_thinking: bool

    @property
    def input_keys(self) -> List[str]:
        return ["filtered_reviews"]

    @property
    def output_keys(self) -> List[str]:
        return ["batch_summaries"]

    def __init__(
        self,
        llm: BaseLanguageModel,
        output_parser: StructuredOutputParser,
        prompt_template: str = summarization_prompts.JUICE_SUMMARIZATION_PROMPT,
        batch_size: int = 12,
        enable_thinking: bool = False,
    ):
        super().__init__(
            llm=llm,
            output_parser=output_parser,
            prompt_template=prompt_template,
            batch_size=batch_size,
            enable_thinking=enable_thinking,
        )
        self.llm = llm
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        self.batch_size = batch_size
        self.enable_thinking = enable_thinking

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        reviews = inputs["filtered_reviews"]
        prompt = ChatPromptTemplate(
            [("system", "" if self.enable_thinking else "/no_think"), ("human", self.prompt_template)]
        )
        format_instructions = self.output_parser.get_format_instructions()
        summarization_chain = prompt | self.llm | self.output_parser

        batch_summaries = []
        review_batches = [reviews[i : i + self.batch_size] for i in range(0, len(reviews), self.batch_size)]
        summarization_outputs = summarization_chain.batch(
            [
                {"review_texts": "\n\n".join([review["review"] for review in review_batch]), "format_instructions": format_instructions}
                for review_batch in review_batches
            ]
        )
        assert len(summarization_outputs) == len(review_batches)
        return {"batch_summaries": summarization_outputs}
