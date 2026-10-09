"""CLI for rag_knowledge."""

import argparse
import asyncio
import os
import sys

from . import load_env
from .service import get_rag_service, query_rag


def list_knowledge_entries():
    """List registered knowledge entries."""
    service = get_rag_service()
    docs = service.retriever.documents
    total_str = f"{len(docs)}" if len(docs) < 100 else f"{len(docs)}+ (limit 100 reached)"
    print(f"\n--- Registered Knowledge Documents ({total_str}) ---")
    for doc in docs:
        print(f" * [{doc.get('id')}] {doc.get('title')}")
        print(f"   Keywords: {', '.join(doc.get('keywords', []))}")
        print(f"   Summary:  {doc.get('summary')}")
        print()


async def run_query(query: str, verbose: bool = True):
    """Execute a query against the RAG service and display results."""
    service = get_rag_service()
    pre_matches = None
    cli_top_k = int(os.getenv("RAG_TOP_K", "5"))
    if verbose:
        pre_matches = await asyncio.to_thread(service.retriever.retrieve, query, top_k=cli_top_k)
        print(f"\n[Retrieval Matches for '{query}']:")
        if pre_matches:
            for idx, m in enumerate(pre_matches, 1):
                print(f"  {idx}. {m.get('title')} (score={m.get('score')})")
        else:
            print("  (No documents met the relevance threshold)")

    print(f"\n[Query]: {query}")
    answer = await service.query(query, pre_retrieved=pre_matches)
    print(f"\n[Answer]:\n{answer}\n")


def main():
    load_env()
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Base - Standalone CLI & Retrieval Tool"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query or question (e.g. 'What events are scheduled?')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all documents in the knowledge base",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Print retrieval scoring and matching details (enabled by default)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress retrieval ranking matches and print only the final answer",
    )

    args = parser.parse_args()

    if args.list:
        list_knowledge_entries()
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    show_matches = False if args.quiet else True
    asyncio.run(run_query(args.query, verbose=show_matches))


if __name__ == "__main__":
    main()
