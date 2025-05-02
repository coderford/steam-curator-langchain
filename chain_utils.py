from langchain.chains.sequential import SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.exceptions import OutputParserException

import output_parsers
import model_registry
from chains import filter_chains, summarization_chains, aggregation_chains
from prompts import filter_prompts, summarization_prompts, aggregation_prompts


def get_language_model(model, temperature=0.7):
    LLMClass = model_registry.LLM_CLASS_MAP.get(model)
    if LLMClass is None:
        raise ValueError(f"Unrecognized language model: {model}")
    return LLMClass(model=model, temperature=temperature)


def get_blurb(review_text, model="qwen2.5:7b", temperature=0.7):
    """
    Generates a blurb for a given review text using the specified language model.

    Args:
        review_text (str): The review text to generate a blurb for.
        model (str, optional): The language model to use for generating the blurb. Defaults to "qwen2.5:7b".

    Returns:
        str: The generated blurb.
    """
    blurb_prompt = ChatPromptTemplate.from_template(aggregation_prompts.BLURB_PROMPT)
    llm = get_language_model(model=model, temperature=temperature)

    chain = blurb_prompt | llm | StrOutputParser()
    output = chain.invoke({"review_text": review_text})
    return output


def get_filter_chain(model, temperature=0.7):
    """
    Creates a filter chain to determine whether reviews should be included based on certain criteria.

    Args:
        model (str): The language model to use for filtering.
        temperature (float, optional): The temperature parameter for the language model. Defaults to 0.0.

    Returns:
        Chain: A LangChain chain that includes deterministic filtering and LLM-based filtering.
    """
    deterministic_filter = filter_chains.DeterministicFilterChain()

    filter_llm = get_language_model(model=model, temperature=temperature)
    llm_filter = filter_chains.LLMFilterChain(
        filter_llm,
        output_parser=output_parsers.FILTER_CHAIN_PARSER,
        prompt_template=filter_prompts.FLUFF_FILTER_PROMPT,
    )

    remap_output = RunnableLambda(lambda x: {"remapped_reviews": x["filtered_reviews"]})
    remap_input = RunnableLambda(lambda x: {"reviews": x["remapped_reviews"]})

    filter_chain = deterministic_filter | remap_output | remap_input | llm_filter
    return filter_chain


def get_summarization_chain(model, temperature=0.7, batch_size=12):
    """
    Creates a summarization chain to generate summaries of the filtered reviews.

    Args:
        model (str): The language model to use for summarization.
        temperature (float, optional): The temperature parameter for the language model. Defaults to 0.0.

    Returns:
        Chain: A LangChain chain that performs LLM-based summarization of reviews.
    """
    summary_llm = get_language_model(model=model, temperature=temperature)
    summarization_chain = summarization_chains.SummarizationChain(
        summary_llm,
        output_parser=output_parsers.JUICE_SUMMARIZATION_CHAIN_PARSER,
        prompt_template=summarization_prompts.JUICE_SUMMARIZATION_PROMPT,
        batch_size=batch_size,
    )
    return summarization_chain


def get_aggregation_chain(model, temperature=0.7, num_retries=2):
    """
    Creates an aggregation chain to generate summaries of the filtered reviews for each aspect.

    Args:
        model (str): The language model to use for aggregation.
        temperature (float, optional): The temperature parameter for the language model. Defaults to 0.0.

    Returns:
        Chain: A LangChain chain that performs LLM-based aggregation of review summaries and output parsing.
    """
    aggregation_llm = get_language_model(model=model, temperature=temperature)
    aspects = list(aggregation_prompts.JUICE_AGGREGATION_PROMPTS.keys())
    aggregation_branches = {}
    for aspect in aspects:
        remap_input = RunnableLambda(
            (lambda y: (lambda x: {"batch_summaries": x["batch_summaries"], "summary_aspect": y}))(aspect)
        )
        chain = aggregation_chains.AggregationChain(
            aggregation_llm,
            output_parser=output_parsers.JUICE_AGGREGATION_CHAIN_PARSER,
            prompt_template=aggregation_prompts.JUICE_AGGREGATION_PROMPTS[aspect],
        ).with_retry(stop_after_attempt=num_retries, retry_if_exception_type=[OutputParserException])
        aggregation_branches[aspect] = remap_input | chain

    aggregation_chain = RunnableParallel(branches=aggregation_branches)
    return aggregation_chain


def make_complete_chain(
    filter_model="gemma3:4b",
    summarization_model="qwen2.5:7b",
    aggregation_model="gemma3:12b",
    summarization_batch_size=12,
    temperature=0.0,
):
    """
    Creates a complete chain that filters, summarizes, and aggregates reviews using specified language models.

    Args:
        filter_model (str): The language model to use for filtering reviews.
        summarization_model (str): The language model to use for summarizing reviews.
        aggregation_model (str): The language model to use for aggregating review summaries.

    Returns:
        Chain: A LangChain chain that filters, summarizes, and aggregates reviews based on the specified models.
    """
    filter_chain = get_filter_chain(filter_model, temperature=temperature)
    summarization_chain = get_summarization_chain(summarization_model, temperature=temperature, batch_size=summarization_batch_size)
    aggregation_chain = get_aggregation_chain(aggregation_model, temperature=temperature)
    complete_chain = filter_chain | summarization_chain | aggregation_chain
    return complete_chain
