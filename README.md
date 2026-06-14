# Building AI Agents from Scratch with Python: Chatbots, RAG, and LangChain

This project is a hands-on lab that walks through building AI agents and retrieval-augmented systems from the ground up in Python. It combines simple interactive chat loops, GPT‑2 generation with DPR retrieval, PyTorch-based RAG, and LangChain-powered tools for working with documents and CSV data. The goal is to understand how agents actually work beneath the frameworks, and how retrieval, generation, and prompting come together. [web:50][web:55]

---

## Features

- **Interactive console chatbot**: “The Daily Dish Chatbot” with a simple Python loop and extensible response logic.
- **RAG with GPT‑2 + DPR**: Compare direct GPT‑2 answers vs. answers augmented with Dense Passage Retrieval contexts.
- **PyTorch RAG with cosine similarity**: Implement and swap out similarity metrics in a custom `RAG_QA` function.
- **In-context engineering**: Explore how temperature, top‑p, and max tokens shape an LLM’s behavior and build a one‑shot text classification agent.
- **LangChain exercises**: Split documents into chunks with custom separators and build a CSV agent that can “talk” to tabular data like `student-mat.csv`. [web:54][web:57]

---

## Project Structure

All files live at the project root (no folders required):

- `daily_dish_chatbot.py` – Interactive chat loop for a simple restaurant-style chatbot.
- `gpt2_dpr_rag.py` – GPT‑2 answers with and without DPR-retrieved contexts (RAG-style).
- `rag_pytorch_cosine.py` – RAG QA function implemented in PyTorch using cosine similarity.
- `incontext_prompt_experiments.py` – Prompt engineering and one‑shot classification exercises with LangChain.
- `langchain_exercises.py` – Document splitting and CSV agent exercises using LangChain.
- `README.md` – Project documentation.

You can optionally add `student-mat.csv` (UCI student performance dataset) as a CSV file in the root directory for the CSV agent exercise.

---

## Installation

Create and activate a virtual environment (recommended), then install dependencies:

```bash
pip install torch transformers sentence-transformers \
    langchain langchain-community faiss-cpu \
    pandas numpy scikit-learn accelerate
```

This setup covers GPT‑2 and DPR from Hugging Face, vector similarity with FAISS, LangChain tooling, and basic data analysis libraries. [web:55][web:58]

If you use Hugging Face Hub models (e.g., Llama 3) in the LangChain scripts, set your token:

```bash
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

On Windows PowerShell:

```powershell
$env:HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

---

## 1. The Daily Dish Chatbot

**File:** `daily_dish_chatbot.py`

Run:

```bash
python daily_dish_chatbot.py
```

You’ll see:

```text
🍽️ Welcome to The Daily Dish Chatbot!
Type 'exit' to end the conversation.
```

Type messages like:

- `What’s on the menu today?`
- `Do you have vegan options?`
- `Can you recommend something?`

Type `exit`, `quit`, or `bye` to end the chat. This file demonstrates an interactive loop and gives you a simple place to swap in an LLM later.

---

## 2. Enhancing GPT‑2 with DPR (RAG-style)

**File:** `gpt2_dpr_rag.py`  

This script compares:

- **Direct generation** – GPT‑2 answers a question using only its pretrained knowledge.
- **RAG generation** – GPT‑2 receives retrieved DPR contexts as additional input, improving specificity and relevance.

You will see three outputs:

1. Direct GPT‑2 answer (no retrieval).
2. RAG answer with default generation parameters.
3. RAG answer with tuned parameters (e.g., larger `num_beams`, higher `max_length`).

### Parameter tuning exercise

Inside `generate_answer`, experiment with:

- `max_length` and `min_length` – How long the answer can be.
- `num_beams` – Beam search width for more exhaustive decoding.
- `length_penalty` – Encourages shorter or longer answers.
- `temperature` and `top_p` – Control creativity vs. determinism.

Observe how different settings change conciseness, faithfulness to context, and overall quality of the response. This mirrors common RAG tuning workflows described in RAG tutorials. [web:55][web:58]

Run:

```bash
python gpt2_dpr_rag.py
```

---

## 3. RAG with PyTorch and Cosine Similarity

**File:** `rag_pytorch_cosine.py`  

This script implements a simplified RAG-style QA:

- You have embeddings for prior questions and responses.
- You compute similarity between a new query embedding and stored response embeddings.
- You retrieve the most relevant responses. [web:43]

The exercise is to replace the dot-product similarity with **cosine similarity**, which focuses on vector orientation rather than magnitude. The file:

- Computes norms for embeddings.
- Builds a cosine similarity matrix.
- Sorts responses by similarity and returns the top‑k matches.

Run:

```bash
python rag_pytorch_cosine.py
```

You will see the top responses with their cosine scores.

---

## 4. In-Context Engineering and Prompt Templates

**File:** `incontext_prompt_experiments.py`  

This module covers three exercises:

1. **Change LLM parameters** – Build an LLM with custom `max_new_tokens`, `temperature`, and `top_p` and see how outputs change.
2. **Observe how the LLM thinks** – Use `verbose=True` in `LLMChain` to inspect intermediate reasoning and prompts.
3. **Zero-shot to one-shot classification** – Upgrade a zero‑shot text classifier to one‑shot by adding a single labeled example in the prompt. [web:31][web:46]

Run:

```bash
python incontext_prompt_experiments.py
```

It will perform a simple one‑shot classification over a short student-performance text. You can adjust the example, labels, and parameters to see how behavior changes.

---

## 5. LangChain Document & CSV Exercises

**File:** `langchain_exercises.py`  

This file contains two main exercises:

1. **Document splitting with custom separators**  
   Uses `RecursiveCharacterTextSplitter` with separators like `["\n\n", "\n", ". ", " ", ""]` to show how different splitting strategies change chunk boundaries. Useful for RAG and chunk-level context management.

2. **CSV agent for data QA**  
   Uses a LangChain CSV agent to load a CSV (e.g., `student-mat.csv`) and answer questions like “What is the average final grade (G3)?” or “How many students have higher than average absences?”. [web:54][web:57]

To run:

```bash
python langchain_exercises.py
```

