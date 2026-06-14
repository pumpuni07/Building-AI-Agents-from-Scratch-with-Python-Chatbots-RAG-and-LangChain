"""
LangChain Exercises

- Exercise 2: Split a document with different separators.
- Exercise 3: Create an agent to talk with CSV data.
"""

from pathlib import Path
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.agent_toolkits import create_csv_agent


def split_document_with_separators(text: str):
    """
    Exercise 2: Using custom separators for splitting.
    Example separators: ["\\n\\n", "\\n", ". ", " ", ""]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks.")
    for i, ch in enumerate(chunks[:5]):
        print(f"\n---- Chunk {i} ----\n{ch[:200]}")
    return chunks


def build_csv_agent(csv_path: str):
    """
    Exercise 3: Agent to talk with CSV data.
    Uses LangChain CSV agent toolkit.
    """
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"temperature": 0.2, "max_new_tokens": 256},
    )

    agent = create_csv_agent(
        llm=llm,
        path=csv_path,
        verbose=True,
    )
    return agent


if __name__ == "__main__":
    # Example text for splitting
    sample_text = (
        "This is a sample policy document.\n\n"
        "Smoking is prohibited in all indoor areas.\n"
        "Employees must comply with safety guidelines. "
        "Failure to do so may result in disciplinary action.\n\n"
        "Thank you for your cooperation."
    )

    print("=== Exercise 2: Split document ===")
    split_document_with_separators(sample_text)

    # Example CSV agent with student-mat CSV
    print("\n=== Exercise 3: CSV Agent ===")
    csv_file = "student-mat.csv"  # save your CSV (from paste) under this name in the project root
    if Path(csv_file).exists():
        agent = build_csv_agent(csv_file)
        answer = agent.run("What is the average final grade (G3) for this dataset?")
        print(answer)
    else:
        print(f"CSV file '{csv_file}' not found. Please download and save it in the same directory.")
