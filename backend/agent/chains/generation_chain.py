"""Final answer generation, grounded only in provided context, with citations."""

from langchain_core.documents import Document

from backend.agent.llm_provider import get_chat_llm

GENERATION_PROMPT = (
    "You are an assistant answering questions using ONLY the provided context. "
    "Always cite your sources inline using the format [source: filename]. "
    'If the context does not contain enough information to answer, respond exactly with: '
    '"I don\'t know based on the available information."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def generate_answer(question: str, documents: list[Document]) -> str:
    llm = get_chat_llm(tier="flash_lite", temperature=0.2)

    context = "\n\n".join(
        f"[source: {doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
        for doc in documents
    )
    if not context:
        context = "(no relevant context found)"

    prompt = GENERATION_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content.strip()
