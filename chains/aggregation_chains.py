from typing import List, Dict, Any

from langchain.chains.base import Chain
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

class AggregationChain(Chain):
    llm: BaseLanguageModel
    prompt_template: str
    output_parser: StructuredOutputParser
    enable_thinking: bool

    @property
    def input_keys(self) -> List[str]:
        return ["batch_summaries", "summary_aspect"]

    @property
    def output_keys(self) -> List[str]:
        return ["aggregate_score", "score_explanation"]

    def __init__(
        self,
        llm: BaseLanguageModel,
        output_parser: StructuredOutputParser,
        prompt_template: str,
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
        summary_aspect = inputs["summary_aspect"]
        summaries = [x[summary_aspect] for x in inputs["batch_summaries"]]
        prompt = ChatPromptTemplate([
            ("system", "" if self.enable_thinking else "/no_think"),
            ("human", self.prompt_template)
        ])
        format_instructions = self.output_parser.get_format_instructions()
        chain = prompt | self.llm | self.output_parser
        
        return chain.invoke(
            {"summary_texts": '\n\n'.join(summaries), "format_instructions": format_instructions}
        )
