from langchain_community.chat_models.openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings


THINKING_MODELS = [
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
]

LLM_CLASS_MAP = {
    # Ollama models
    "gemma3:4b": ChatOllama,
    "gemma3:12b": ChatOllama,
    "qwen3:1.7b": ChatOllama,
    "qwen3:4b": ChatOllama,
    "qwen3:8b": ChatOllama,
    "qwen3:14b": ChatOllama,
    "qwen2.5:7b": ChatOllama,
    "granite3.3:8b": ChatOllama,

    # OpenAI models
    "gpt-4o": ChatOpenAI,
    "gpt-4o-mini": ChatOpenAI,
    "gpt-4.1": ChatOpenAI,
    "gpt-4.1-mini": ChatOpenAI,

    # Google models
    "gemini-2.0-flash": ChatGoogleGenerativeAI,
    "gemini-2.0-flash-lite": ChatGoogleGenerativeAI,
}

EMBEDDING_CLASS_MAP = {
    # Google models
    "models/text-embedding-004": GoogleGenerativeAIEmbeddings,
    "models/gemini-embedding-exp": GoogleGenerativeAIEmbeddings,

    # Ollama models
    "nomic-embed-text:latest": OllamaEmbeddings,
    "snowflake-arctic-embed2": OllamaEmbeddings,
}