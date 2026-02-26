"""
Sage People Brain — Pinecone connection layer.

Handles creating the index and uploading/querying vectors.
You don't need to edit this file directly.
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "sage-people"
EMBEDDING_DIMENSION = 1536  # matches text-embedding-3-small
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


def get_index():
    """Connect to Pinecone and return the index (creates it if it doesn't exist)."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait until the index is ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
        print("  Index created and ready.")

    return pc.Index(INDEX_NAME)


def upsert_vectors(vectors: list) -> int:
    """Upload a list of vectors to Pinecone in batches of 100."""
    index = get_index()
    batch_size = 100
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        total += len(batch)
    return total


def search(query_embedding: list, top_k: int = 5, filter: dict = None) -> list:
    """Search Pinecone with a query embedding. Returns a list of matches."""
    index = get_index()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
    )
    return results.matches


def get_index_stats() -> dict:
    """Return stats about the index (useful for checking what's uploaded)."""
    index = get_index()
    return index.describe_index_stats()
