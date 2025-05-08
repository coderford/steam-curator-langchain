# LangChain Steam Game Reviewer and QA

An LLM-based steam game reviewer that parses through hundreds of steam reviews to form an opinion. Generates a "Juice Score" based on various aspects like atmosphere, exploration, gameplay mechanics, challenge etc. FromSoftware Games will usually receive high scores.

Supports local models via Ollama and Gemini/OpenAI models as well (you'll need to set `GOOGLE_API_KEY` or `OPENAI_API_KEY` in your env).
Also includes a command-line QA interface powered by Retrieval-Augmented Generation (RAG), allowing you to ask specific questions and receive answers based on the review data.

## Usage

Run `run_chains.py` to get Juice Score and detailed explanations for a given Steam app ID:
```sh
export GOOGLE_API_KEY=<your google gemini API key>
python3 run_chains.py --app_id 1245620 # Get Juice score for Elden Ring
```

Run `rag_qa.py` to set up an interactive QA interface based on reviews of a given game:
```sh
export GOOGLE_API_KEY=<your google gemini API key>
python3 rag_qa.py -i 1245620
```

