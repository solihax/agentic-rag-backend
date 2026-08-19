"""
Usage:
    python -m backend.scripts.evaluate
"""

import time

from backend.agent.chains.graders import grade_answer_usefulness, grade_hallucination
from backend.agent.graph.graph import build_graph

EVAL_QUESTIONS = [
    {"question": "What is this document about?", "expect_keyword": "sample"},
]


def run_evaluation() -> None:
    graph = build_graph()

    results = []
    for case in EVAL_QUESTIONS:
        question = case["question"]
        start = time.perf_counter()

        result = graph.invoke(
            {
                "question": question,
                "generation": "",
                "documents": [],
                "generation_retries": 0,
                "web_search_retries": 0,
            }
        )

        latency = time.perf_counter() - start
        documents = result["documents"]
        generation = result["generation"]

        hit = any(
            case["expect_keyword"].lower() in doc.page_content.lower()
            for doc in documents
        )

        documents_text = "\n\n".join(doc.page_content for doc in documents)
        grounded = grade_hallucination(documents_text, generation) if documents_text else None
        useful = grade_answer_usefulness(question, generation)

        results.append(
            {
                "question": question,
                "latency_sec": round(latency, 2),
                "retrieval_hit": hit,
                "grounded": grounded,
                "useful": useful,
                "num_sources": len(documents),
            }
        )

    print(f"{'Question':<40} {'Hit':<6} {'Grounded':<10} {'Useful':<8} {'Latency(s)':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['question'][:38]:<40} "
            f"{str(r['retrieval_hit']):<6} "
            f"{str(r['grounded']):<10} "
            f"{str(r['useful']):<8} "
            f"{r['latency_sec']:<10}"
        )

    hit_rate = sum(r["retrieval_hit"] for r in results) / len(results)
    grounded_rate = sum(1 for r in results if r["grounded"]) / len(results)
    useful_rate = sum(1 for r in results if r["useful"]) / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)

    print("\nSummary:")
    print(f"  Retrieval hit rate: {hit_rate:.0%}")
    print(f"  Groundedness rate:  {grounded_rate:.0%}")
    print(f"  Usefulness rate:    {useful_rate:.0%}")
    print(f"  Avg latency:        {avg_latency:.2f}s")


if __name__ == "__main__":
    run_evaluation()