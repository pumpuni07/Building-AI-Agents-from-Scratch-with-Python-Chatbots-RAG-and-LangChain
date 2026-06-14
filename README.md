# Lab: Building AI Agents from Scratch with Python

This lab implements several hands-on exercises for building AI agents and retrieval-augmented generation (RAG) systems in Python:

- Interactive console chatbot (`The Daily Dish Chatbot`)
- Enhancing GPT-2 answers using DPR retrieval
- RAG-style QA in PyTorch with cosine similarity
- In-context engineering and prompt templates
- LangChain exercises for document splitting and CSV QA

## Files

- `daily_dish_chatbot.py` – Simple interactive console chatbot.
- `gpt2_dpr_rag.py` – GPT-2 generation with and without DPR contexts.
- `rag_pytorch_cosine.py` – RAG-style QA using cosine similarity.
- `incontext_prompt_experiments.py` – In-context engineering and prompt templates.
- `langchain_exercises.py` – LangChain tasks (split documents, CSV agent).
- `README.md` – This documentation.

## Getting Started

Create a virtual environment and install requirements (adapt to your environment):

```bash
pip install torch transformers sentence-transformers langchain langchain-community faiss-cpu pandas
```

You will also need internet access to download models like GPT-2 and DPR encoders from Hugging Face. [web:35][web:46]

Each script can be run independently, for example:

```bash
python daily_dish_chatbot.py
python gpt2_dpr_rag.py
python rag_pytorch_cosine.py
```

Refer to comments in each file for specific instructions on the exercises.
