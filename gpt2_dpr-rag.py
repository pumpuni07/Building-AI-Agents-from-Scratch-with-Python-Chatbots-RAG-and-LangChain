"""
Enhance LLMs using RAG and Hugging Face (GPT-2 + DPR)

- Direct generation with GPT-2
- Generation with DPR-retrieved contexts (RAG-style)
- Exercise: tune generation parameters (max_length, min_length, length_penalty, num_beams)
"""

from typing import List
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizerFast,
    DPRContextEncoderTokenizerFast,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    # GPT-2
    gpt2_model_name = "gpt2"
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_name)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(DEVICE)

    # DPR
    q_model_name = "facebook/dpr-question_encoder-single-nq-base"
    c_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
    q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(q_model_name)
    q_encoder = DPRQuestionEncoder.from_pretrained(q_model_name).to(DEVICE)
    c_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(c_model_name)
    c_encoder = DPRContextEncoder.from_pretrained(c_model_name).to(DEVICE)

    return gpt2_model, gpt2_tokenizer, q_encoder, q_tokenizer, c_encoder, c_tokenizer


def encode_contexts(contexts: List[str], c_encoder, c_tokenizer):
    inputs = c_tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        embeddings = c_encoder(**inputs).pooler_output
    return embeddings  # shape: (num_contexts, dim)


def encode_query(query: str, q_encoder, q_tokenizer):
    inputs = q_tokenizer(query, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        embedding = q_encoder(**inputs).pooler_output
    return embedding  # shape: (1, dim)


def retrieve_contexts(query_emb, ctx_embs, contexts: List[str], top_k: int = 3) -> List[str]:
    # Dot product for similarity
    scores = torch.matmul(query_emb, ctx_embs.T).squeeze(0)  # (num_contexts,)
    topk = torch.topk(scores, k=min(top_k, len(contexts))).indices.tolist()
    return [contexts[i] for i in topk]


def generate_answer(
    question: str,
    retrieved_contexts: List[str],
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    max_length: int = 128,
    min_length: int = 32,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """
    Exercise: Tuning generation parameters
    - Try different values of max_length, min_length, length_penalty, and num_beams.
    """
    prompt = "Context:\n" + "\n".join(retrieved_contexts) + "\n\nQuestion: " + question + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated[len(prompt) :].strip()


def generate_answer_direct(
    question: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    **gen_kwargs,
) -> str:
    prompt = "Question: " + question + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated[len(prompt) :].strip()


if __name__ == "__main__":
    gpt2_model, gpt2_tokenizer, q_encoder, q_tokenizer, c_encoder, c_tokenizer = load_models()

    # Example contexts for the lab
    contexts = [
        "Our smoking policy prohibits smoking in all indoor areas and within 10 meters of entrances.",
        "Children under 16 must be accompanied by an adult in all facilities.",
        "The cafeteria serves hot meals from 11:00 to 14:00 on weekdays.",
    ]

    ctx_embs = encode_contexts(contexts, c_encoder, c_tokenizer)

    question = "What is the smoking policy?"

    # Direct generation (no DPR contexts)
    direct_answer = generate_answer_direct(
        question,
        gpt2_model,
        gpt2_tokenizer,
        max_length=80,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        num_beams=1,
    )

    # With DPR contexts (RAG-style)
    q_emb = encode_query(question, q_encoder, q_tokenizer)
    retrieved = retrieve_contexts(q_emb, ctx_embs, contexts, top_k=2)
    rag_answer_default = generate_answer(
        question,
        retrieved,
        gpt2_model,
        gpt2_tokenizer,
        max_length=80,
        min_length=30,
        num_beams=4,
        length_penalty=1.0,
    )

    # Exercise: try different parameter sets
    rag_answer_beamy = generate_answer(
        question,
        retrieved,
        gpt2_model,
        gpt2_tokenizer,
        max_length=120,
        min_length=40,
        num_beams=8,
        length_penalty=1.5,
    )

    print("=== Direct GPT-2 (no DPR) ===")
    print(direct_answer)
    print("\n=== RAG (DPR + GPT-2, default params) ===")
    print(rag_answer_default)
    print("\n=== RAG (DPR + GPT-2, tuned params) ===")
    print(rag_answer_beamy)
