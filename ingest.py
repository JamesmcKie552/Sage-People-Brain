#!/usr/bin/env python3
"""
Sage People Brain — PDF Ingestion Pipeline

Reads PDFs from the docs/ folder, extracts text, chunks it, uses Claude
to auto-tag metadata, generates embeddings, and uploads to Pinecone.

Usage:
    python ingest.py                          # Process all PDFs
    python ingest.py --dry-run               # Preview without uploading
    python ingest.py --folder "Case Studies" # Process one folder only

Folder structure expected inside docs/:
    docs/
    ├── Case Studies/
    ├── ICP & Personas/
    ├── Messaging & Positioning/
    ├── Product Docs/
    ├── Reports & Research/
    └── Competitor Intel/
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import re
import fitz  # PyMuPDF — reads PDFs
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

import vector_store as vs

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

DOCS_DIR = Path(".")  # folders sit in the repo root
CHUNK_SIZE = 800     # tokens per chunk (roughly 600 words)
CHUNK_OVERLAP = 150  # tokens of overlap between chunks so context isn't lost

# Maps each subfolder name to the doc_type metadata value
FOLDER_TO_DOC_TYPE = {
    "Case Studies (2)":   "case_study",
    "Competitor Analysis": "competitor",
    "ICP & Personas":     "icp_persona",
    "Market Context":     "market_context",
    "Messaging":          "messaging",
    "Product Information": "product",
    "Reports & Content":  "report",
}

# ─── API Clients ──────────────────────────────────────────────────────────────

def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"{key} not found in .env file")
    return val

openai_client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))
claude_client  = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
tokenizer      = tiktoken.get_encoding("cl100k_base")

# ─── Step 1: Extract text from PDF ────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    """Extract and clean text from a PDF. Returns empty string if PDF is image-based."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    raw = "\n".join(pages)
    return _clean_text(raw)


def _clean_text(text: str) -> str:
    """Fix common PDF extraction artefacts: broken lines, hyphenation, extra whitespace."""
    # Rejoin words hyphenated across line breaks (e.g. "transfor-\nmation" → "transformation")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Replace single newlines (mid-paragraph) with a space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse runs of spaces
    text = re.sub(r" {2,}", " ", text)
    # Reduce 3+ blank lines to one paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ─── Step 2: Split text into overlapping chunks ───────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into token-based chunks with overlap."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ─── Step 3: Claude auto-tags metadata ────────────────────────────────────────

def extract_metadata(text: str, doc_type: str, filename: str) -> dict:
    """
    Ask Claude to read the document and suggest metadata.
    We only send the first ~3,000 words to keep costs low.
    """
    preview = " ".join(text.split()[:3000])

    prompt = f"""You are analysing a Sage People (cloud HR/HCM software) document to extract metadata for a sales knowledge base.

Document filename: {filename}
Document type: {doc_type}

Document content:
{preview}

Return ONLY a valid JSON object with these exact fields — no other text, no markdown:
{{
    "segment": "enterprise" or "mid_market" or "all",
    "pain_points": ["2-5 specific HR pain points mentioned or implied in this document"],
    "persona": "the primary job title this document targets (e.g. CHRO, HR Director, HR Manager, CFO, IT Director, all)"
}}

Guidelines:
- segment: use "enterprise" for 1000+ employees, "mid_market" for 200-999, "all" if not specified
- pain_points: be specific, e.g. "manual payroll processing", "lack of workforce visibility", "compliance risk", "high HR admin burden", "poor employee experience"
- persona: pick the single most relevant job title, or "all" if it's general"""

    message = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if Claude returns them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ⚠️  Could not parse Claude's response for {filename}, using defaults")
        return {"segment": "all", "pain_points": [], "persona": "all"}


def extract_metadata_safe(text: str, doc_type: str, filename: str) -> dict:
    """Wrapper that falls back to defaults if Claude API is unavailable."""
    try:
        return extract_metadata(text, doc_type, filename)
    except Exception as e:
        print(f"  ⚠️  Claude unavailable ({type(e).__name__}), using default metadata")
        return {"segment": "all", "pain_points": [], "persona": "all"}

# ─── Step 4: Generate embedding ───────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """Convert text to a vector using OpenAI's embedding model."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

# ─── Main: Process a single PDF ───────────────────────────────────────────────

def ingest_document(pdf_path: Path, doc_type: str, dry_run: bool = False) -> int:
    """Full pipeline for one PDF: extract → chunk → tag → embed → upload."""
    filename = pdf_path.name

    print(f"\n📄  {filename}")

    # 1. Extract
    text = extract_text(pdf_path)
    if not text:
        print(f"  ⚠️  No text found — PDF may be image-based, skipping")
        return 0
    print(f"  ✅  Extracted {len(text.split()):,} words")

    # 2. Claude metadata (once per document — not per chunk)
    print(f"  🤖  Asking Claude to tag metadata...")
    meta = extract_metadata_safe(text, doc_type, filename)
    print(f"  ✅  segment={meta['segment']}  persona={meta['persona']}")
    if meta["pain_points"]:
        print(f"       pain points: {', '.join(meta['pain_points'][:3])}")

    # 3. Chunk
    chunks = chunk_text(text)
    print(f"  ✂️   Split into {len(chunks)} chunk(s)")

    if dry_run:
        print(f"  🔍  DRY RUN — would upload {len(chunks)} vector(s)")
        return len(chunks)

    # 4. Embed + build vector records
    vectors = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{filename}_chunk{i}".encode()).hexdigest()
        embedding = get_embedding(chunk)
        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": {
                "doc_type":    doc_type,
                "segment":     meta["segment"],
                "pain_points": meta["pain_points"],
                "persona":     meta["persona"],
                "file_name":   filename,
                "file_path":   str(pdf_path),
                "chunk_index": i,
                "text":        chunk,   # stored so we can read it back on retrieval
            },
        })

    # 5. Upload
    vs.upsert_vectors(vectors)
    print(f"  ✅  Uploaded {len(vectors)} vector(s) to Pinecone")
    return len(vectors)

# ─── Entry point ──────────────────────────────────────────────────────────────

def run(folder_filter: Optional[str] = None, dry_run: bool = False):
    if not DOCS_DIR.exists():
        print("❌  docs/ folder not found.\n")
        print("Create it with this structure and add your PDFs:")
        for folder in FOLDER_TO_DOC_TYPE:
            print(f"    docs/{folder}/")
        return

    # Collect all PDFs across folders
    to_process = []
    for folder_name, doc_type in FOLDER_TO_DOC_TYPE.items():
        if folder_filter and folder_name != folder_filter:
            continue
        folder_path = DOCS_DIR / folder_name
        if not folder_path.exists():
            continue
        for pdf in sorted(folder_path.glob("*.pdf")):
            to_process.append((pdf, doc_type))

    if not to_process:
        print("❌  No PDFs found. Check your docs/ folder structure.")
        return

    label = "DRY RUN — " if dry_run else ""
    print(f"\n{label}Found {len(to_process)} PDF(s) to process")
    print("=" * 60)

    total_vectors = 0
    for pdf_path, doc_type in to_process:
        count = ingest_document(pdf_path, doc_type, dry_run=dry_run)
        total_vectors += count

    print("\n" + "=" * 60)
    action = "Would upload" if dry_run else "Uploaded"
    print(f"✅  Done! {action} {total_vectors:,} vector(s) from {len(to_process)} document(s)")

    if not dry_run:
        stats = vs.get_index_stats()
        total_in_index = stats.get("total_vector_count", "?")
        print(f"📊  Total vectors now in Pinecone: {total_in_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Sage People PDFs into Pinecone"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without uploading",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help='Only process one folder, e.g. --folder "Case Studies"',
    )
    args = parser.parse_args()
    run(folder_filter=args.folder, dry_run=args.dry_run)
