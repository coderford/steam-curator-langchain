from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema

FILTER_CHAIN_SCHEMAS = [
    ResponseSchema(name="keep_review", description="Boolean true/false, indicates whether to keep the review or not based on your analysis."),
    ResponseSchema(name="explanation", description="Short explanation for your decision"),
]
FILTER_CHAIN_PARSER = StructuredOutputParser.from_response_schemas(FILTER_CHAIN_SCHEMAS)

SUMMARIZATION_CHAIN_SCHEMAS = [
    ResponseSchema(
        name="summary",
        description="Summary of the review texts in concise, professional language.",
    ),
]
SUMMARIZATION_CHAIN_PARSER = StructuredOutputParser.from_response_schemas(SUMMARIZATION_CHAIN_SCHEMAS)
