# Sage People Brain

A Pinecone vector database knowledge base for Sage People (cloud HR/HCM software).
Reads PDFs, auto-tags metadata using Claude, and enables semantic search across your documents.

---

## What's in here

| File | What it does |
|------|-------------|
| `ingest.py` | Reads PDFs → extracts text → asks Claude to tag metadata → uploads to Pinecone |
| `search.py` | Query the knowledge base to test retrieval |
| `vector_store.py` | Pinecone connection layer (you don't need to edit this) |
| `requirements.txt` | Python packages needed |
| `.env` | Your API keys (never committed to git) |

---

## First-time setup

### 1. Install Python packages

Open a terminal and run:

```bash
pip install -r requirements.txt
```

### 2. Add your PDFs

Create a `docs/` folder with this exact structure and drop your PDFs in:

```
docs/
├── Case Studies/
├── ICP & Personas/
├── Messaging & Positioning/
├── Product Docs/
├── Reports & Research/
└── Competitor Intel/
```

The folder name determines the `doc_type` metadata automatically.

### 3. Test with a dry run first

This shows you what *would* be processed without uploading anything:

```bash
python ingest.py --dry-run
```

### 4. Run the full ingestion

```bash
python ingest.py
```

For each PDF, the script will:
1. Extract the text
2. Ask Claude to suggest: segment, pain points, and persona
3. Split into chunks
4. Upload to Pinecone with metadata

### 5. Test your search

```bash
# Basic search
python search.py "HR challenges for enterprise companies"

# Filter by doc type
python search.py "CHRO pain points" --doc-type case_study

# Filter by segment
python search.py "payroll compliance" --segment enterprise

# Show how many docs are in the index
python search.py --stats
```

---

## Metadata auto-tagged by Claude

For each document, Claude reads it and suggests:

| Field | What it is | Example values |
|-------|-----------|---------------|
| `doc_type` | Set from the folder name | `case_study`, `icp_persona`, `messaging`, `product`, `report`, `competitor` |
| `segment` | Company size target | `enterprise`, `mid_market`, `all` |
| `pain_points` | HR pains mentioned | `["manual payroll processing", "lack of HR visibility"]` |
| `persona` | Job title targeted | `CHRO`, `HR Director`, `HR Manager`, `all` |
| `file_name` | Original filename | `Forrester_TEI_Report.pdf` |

---

## Tips

- **Image-based PDFs** (scanned docs) won't extract text — convert them to text-based PDFs first
- **Re-ingesting** a document is safe — the same chunk ID is generated each time, so Pinecone will overwrite rather than duplicate
- **Process one folder at a time** to test before running everything:
  ```bash
  python ingest.py --folder "Case Studies"
  ```
