#!/usr/bin/env python3
"""
Sage People Brain — Search Tool

Test that your Pinecone knowledge base is working by running queries against it.

Usage:
    python search.py "HR pain points for enterprise companies"
    python search.py "CHRO persona challenges" --doc-type case_study
    python search.py "payroll compliance" --segment enterprise --top-k 10
    python search.py --stats           # Show index stats (how many docs are uploaded)
"""

import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI

import vector_store as vs

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def search(
    query: str,
    doc_type: str = None,
    segment: str = None,
    top_k: int = 5,
):
    print(f'\n🔍  Query: "{query}"')
    filters_applied = []
    if doc_type:
        filters_applied.append(f"doc_type={doc_type}")
    if segment:
        filters_applied.append(f"segment={segment} or all")
    if filters_applied:
        print(f"    Filters: {', '.join(filters_applied)}")
    print("-" * 60)

    # Build Pinecone filter
    pinecone_filter = {}
    if doc_type:
        pinecone_filter["doc_type"] = {"$eq": doc_type}
    if segment:
        # Return results for the requested segment AND generic "all" docs
        pinecone_filter["segment"] = {"$in": [segment, "all"]}

    query_embedding = get_embedding(query)

    matches = vs.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter=pinecone_filter if pinecone_filter else None,
    )

    if not matches:
        print("No results found.")
        return

    for i, match in enumerate(matches, 1):
        m = match.metadata
        print(f"\n#{i}  Score: {match.score:.3f}")
        print(f"    File:        {m.get('file_name', 'N/A')}")
        print(f"    Type:        {m.get('doc_type', 'N/A')}")
        print(f"    Segment:     {m.get('segment', 'N/A')}")
        print(f"    Persona:     {m.get('persona', 'N/A')}")
        pain = m.get("pain_points", [])
        if pain:
            print(f"    Pain points: {', '.join(pain)}")
        excerpt = m.get("text", "")[:300].replace("\n", " ")
        print(f"    Excerpt:     {excerpt}...")


def show_stats():
    stats = vs.get_index_stats()
    total = stats.get("total_vector_count", 0)
    print(f"\n📊  Index: sage-people")
    print(f"    Total vectors: {total:,}")
    if total == 0:
        print("    (No documents uploaded yet — run ingest.py first)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search the Sage People Brain knowledge base"
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Your search query",
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        choices=["case_study", "competitor", "icp_persona", "market_context", "messaging", "product", "report"],
        help="Filter by document type",
    )
    parser.add_argument(
        "--segment",
        type=str,
        choices=["enterprise", "mid_market"],
        help="Filter by segment",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index stats instead of searching",
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.query:
        search(args.query, doc_type=args.doc_type, segment=args.segment, top_k=args.top_k)
    else:
        parser.print_help()
