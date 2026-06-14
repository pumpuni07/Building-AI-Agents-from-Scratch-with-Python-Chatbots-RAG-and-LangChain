"""
In-Context Engineering and Prompt Templates

Exercises:
1. Change parameters for the LLM (max_new_tokens, temperature, top_p).
2. Observe how LLM 'thinks' using verbose logging.
3. Revise a text classification agent from zero-shot to one-shot.
"""

from typing import List
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def build_llm(
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    # Replace with your model of choice from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    )
    return llm


def text_classification_zero_shot(label_space: List[str]) -> LLMChain:
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are a text classification model.\n"
            "Classes: {labels}\n\n"
            "Text: {text}\n"
            "Label:"
        ),
        partial_variables={"labels": ", ".join(label_space)},
    )
    llm = build_llm()
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
    return chain


def text_classification_one_shot(label_space: List[str]) -> LLMChain:
    """
    Exercise 3: convert zero-shot to one-shot by adding an example.
    """
    example_text = "The student achieved excellent grades and participated in many school activities."
    example_label = "positive"

    template = (
        "You are a text classification model.\n"
        "Classes: {labels}\n\n"
        "Example:\n"
        "Text: {example_text}\n"
        "Label: {example_label}\n\n"
        "Now classify the new text.\n"
        "Text: {text}\n"
        "Label:"
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
        partial_variables={
            "labels": ", ".join(label_space),
            "example_text": example_text,
            "example_label": example_label,
        },
    )

    # verbose=True to observe 'how LLM thinks' (chain logs)
    llm = build_llm()
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    return chain


if __name__ == "__main__":
    labels = ["positive", "negative", "neutral"]

    # Exercise 1: play with parameters
    llm = build_llm(max_new_tokens=64, temperature=0.3, top_p=0.8)

    # Exercise 3: one-shot classification
    chain_one_shot = text_classification_one_shot(labels)
    text = "The student frequently misses classes and rarely completes homework."
    print(chain_one_shot.run(text=text))
