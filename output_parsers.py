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
        description="Based on the reviews, how would you best describe the lore, worldbuilding and atmosphere of the game? Do reviewers make particular note or references to these aspects? Does it feature complex themes?",
    ),
    ResponseSchema(
        name="exploration",
        description="Based on the reviews, would describe this game as having a solid exploration aspect? Do reviewers mention or seem particularly impressed by the game's exploration mechanics? Are the secrets fun to discover if there are any?"
    ),
    ResponseSchema(
        name="gameplay_mechanics",
        description="Based on the reviews, how would you best describe gameplay mechanics of the game? Do reviewers make note of specific mechanics they are impressed by? Does at least some reviewer call the gameplay mechanics 'deep' and 'complex'?"
    ),
    # ResponseSchema(
    #     name="artstyle",
    #     description="Based on the reviews, do you think the game sports an impressive and/or cohesive art style? Do reviewers mention or seem particularly impressed with artistic choices in the game's art style?"
    # ),
    ResponseSchema(
        name="emotional_engagement",
        description="Based on the reviews, does the game seem to have real, mature emotional depth, or is it just common tropes used to touch heartstrings? Or does it not have much to do with emotions at all? Do reviewers at all describe the game as emotionally mature and mention specific scenes that are emotionally impactful? Are there mentions of strongly written and complex characters?"
    ),
    ResponseSchema(
        name="bloat_grinding",
        description="Based on the reviews, does the game have a lot of bloat and grindy mechanics/encounters? Do reviewers mention bloat or grindiness? Ignore and DO NOT MENTION cases of repetitiveness due to challenging gameplay or repeated playthroughs."
    ),
]
JUICE_SUMMARIZATION_CHAIN_PARSER = StructuredOutputParser.from_response_schemas(JUICE_SUMMARIZATION_CHAIN_SCHEMAS)

JUICE_AGGREGATION_CHAIN_SCHEMAS = [
    ResponseSchema(
        name="aggregate_score",
        type="integer",
        description="Aggregate score out of 10, as per the instructions"
    ),
    ResponseSchema(
        name="score_explanation",
        type="string",
        description="Detailed explanation for the aggregate score"
    )
]
JUICE_AGGREGATION_CHAIN_PARSER = StructuredOutputParser.from_response_schemas(JUICE_AGGREGATION_CHAIN_SCHEMAS)
