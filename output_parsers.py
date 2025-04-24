from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema

FILTER_CHAIN_SCHEMAS = [
    ResponseSchema(name="keep_review", description="Boolean true/false, indicates whether to keep the review or not based on your analysis."),
    ResponseSchema(name="explanation", description="Short explanation for your decision"),
]
FILTER_CHAIN_PARSER = StructuredOutputParser.from_response_schemas(FILTER_CHAIN_SCHEMAS)

JUICE_SUMMARIZATION_CHAIN_SCHEMAS = [
    ResponseSchema(
        name="lore_worldbuilding_atmosphere",
        description="Based on the reviews, how would you best describe the lore, worldbuilding and atmosphere of the game? Do reviewers make particular note or references to these aspects?",
    ),
    ResponseSchema(
        name="exploration",
        description="Based on the reviews, would describe this game as having a solid exploration aspect? Do reviewers mention or seem particularly impressed by the game's exploration mechanics?"
    ),
    ResponseSchema(
        name="gameplay_mechanics",
        description="Based on the reviews, how would you best describe gameplay mechanics of the game? Do reviewers make note of specific mechanics they are impressed by?"
    ),
    ResponseSchema(
        name="artstyle",
        description="String answering the following questions. Based on the reviews, do you think the game sports an impressive and/or cohesive art style? Do reviewers mention or seem particularly impressed with artistic choices in the game's art style?"
    ),
    ResponseSchema(
        name="emotional_maturity",
        description="String answering the following questions. Based on the reviews, does the game seem to have real, mature emotional depth, or is it just common tropes used to touch heartstrings? Or does it not have much to do with emotions at all? Do reviewers at all describe the game as emotionally mature and mention specific scenes that are emotionally impactful?"
    ),
]
JUICE_SUMMARIZATION_CHAIN_PARSER = StructuredOutputParser.from_response_schemas(JUICE_SUMMARIZATION_CHAIN_SCHEMAS)
