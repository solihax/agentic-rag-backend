"""
Usage:
    python -m backend.scripts.smoke_test_graph "your question here"
"""

import sys

from backend.agent.graph.graph import build_graph


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m backend.scripts.smoke_test_graph "question"')
        sys.exit(1)

    question = sys.argv[1]
    graph = build_graph()

    result = graph.invoke(
        {
            "question": question,
            "generation": "",
            "documents": [],
            "generation_retries": 0,
            "web_search_retries": 0,
        }
    )

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['generation']}")
    print(f"\nSources used: {len(result['documents'])}")
    for doc in result["documents"]:
        print(f"  - {doc.metadata.get('filename', 'unknown')}")


if __name__ == "__main__":
    main()
