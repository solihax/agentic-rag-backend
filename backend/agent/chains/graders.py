"""Graders: document relevance, hallucination, and answer usefulness."""

from backend.agent.llm_provider import get_chat_llm


def grade_document(question: str, document_text: str) -> bool:
    llm = get_chat_llm(tier="flash_lite")
    prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        f"Retrieved document:\n{document_text}\n\n"
        f"User question: {question}\n\n"
        "If the document contains information relevant to answering the question, "
        "respond with exactly one word: YES. Otherwise respond with exactly one word: NO."
    )
    response = llm.invoke(prompt)
    return response.content.strip().upper().startswith("YES")


def grade_hallucination(documents_text: str, generation: str) -> bool:
    llm = get_chat_llm(tier="flash_lite")
    prompt = (
        "You are a grader assessing whether an answer is grounded in a set of facts.\n"
        f"Facts:\n{documents_text}\n\n"
        f"Answer:\n{generation}\n\n"
        "If the answer is fully supported by the facts, respond with exactly one word: YES. "
        "If it contains claims not supported by the facts, respond with exactly one word: NO."
    )
    response = llm.invoke(prompt)
    return response.content.strip().upper().startswith("YES")


def grade_answer_usefulness(question: str, generation: str) -> bool:
    llm = get_chat_llm(tier="flash_lite")
    prompt = (
        "You are a grader assessing whether an answer actually resolves a question.\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{generation}\n\n"
        "If the answer resolves the question, respond with exactly one word: YES. "
        "Otherwise respond with exactly one word: NO."
    )
    response = llm.invoke(prompt)
    return response.content.strip().upper().startswith("YES")
