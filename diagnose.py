#!/usr/bin/env python3
"""
Diagnostic script — shows everything currently in the Pinecone index.
Run with: python diagnose.py
"""

import os
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
import vector_store as vs

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    return openai_client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

# Use broad queries to surface all indexed files
PROBE_QUERIES = [
    "HR payroll enterprise", "case study customer win",
    "competitor analysis", "ICP persona", "messaging positioning",
    "product features", "market research report", "mid-market workforce"
]

print("Scanning Pinecone index...\n")
seen = {}  # file_name -> {doc_type, segment, persona, chunks}

for q in PROBE_QUERIES:
    matches = vs.search(get_embedding(q), top_k=50)
    for m in matches:
        meta = m.metadata
        fname = meta.get("file_name", "")
        if not fname:
            continue
        if fname not in seen:
            seen[fname] = {
                "doc_type": meta.get("doc_type", "?"),
                "segment":  meta.get("segment", "?"),
                "persona":  meta.get("persona", "?"),
                "chunks":   set()
            }
        seen[fname]["chunks"].add(meta.get("chunk_index", 0))

# Group by doc_type
by_type = defaultdict(list)
for fname, info in seen.items():
    by_type[info["doc_type"]].append((fname, info))

print(f"Found {len(seen)} unique documents in the index:\n")
print(f"{'DOC TYPE':<22} {'FILE NAME':<55} {'SEG':<12} {'CHUNKS'}")
print("-" * 105)

for doc_type in sorted(by_type.keys()):
    for fname, info in sorted(by_type[doc_type]):
        print(f"{doc_type:<22} {fname:<55} {info['segment']:<12} {len(info['chunks'])}")

print(f"\nTotal: {len(seen)} documents")

# Check for expected folders that might be missing
print("\n--- Checking for missing doc types ---")
expected = {"case_study", "competitor", "icp_persona", "market_context", "messaging", "product", "report"}
found_types = set(by_type.keys())
missing = expected - found_types
if missing:
    print(f"⚠️  No documents found for: {', '.join(missing)}")
    print("   Check that the folder name exactly matches what's in ingest.py")
else:
    print("✅  All doc types have at least one document")
