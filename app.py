"""
Sage People Brain — Web App

Run locally:  python app.py
Deployed on:  Render (see render.yaml)
"""

import os
import json
import uuid
import threading
import anthropic
from datetime import date
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

import vector_store as vs

# In-memory job store: job_id -> {"status": "running"|"done"|"error", ...}
_jobs = {}

load_dotenv()

app = Flask(__name__)


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    doc_type = data.get("doc_type")   # optional filter
    segment = data.get("segment")     # optional filter
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Build optional Pinecone filter
    pinecone_filter = {}
    if doc_type:
        pinecone_filter["doc_type"] = {"$eq": doc_type}
    if segment:
        pinecone_filter["segment"] = {"$in": [segment, "all"]}

    query_embedding = get_embedding(query)
    matches = vs.search(
        query_embedding=query_embedding,
        top_k=top_k,
        filter=pinecone_filter if pinecone_filter else None,
    )

    results = []
    for match in matches:
        m = match.metadata
        results.append({
            "score":       round(match.score, 3),
            "file_name":   m.get("file_name", ""),
            "doc_type":    m.get("doc_type", ""),
            "segment":     m.get("segment", ""),
            "persona":     m.get("persona", ""),
            "pain_points": m.get("pain_points", []),
            "text":        m.get("text", ""),
        })

    return jsonify({"results": results})


@app.route("/api/battlecard", methods=["POST"])
def battlecard():
    """Kick off battle card generation in a background thread and return a job ID immediately."""
    data = request.get_json()
    competitor = data.get("competitor", "").strip()

    if not competitor:
        return jsonify({"error": "No competitor provided"}), 400

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "step": "Starting…"}

    thread = threading.Thread(
        target=_run_battlecard, args=(job_id, competitor), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/battlecard/status/<job_id>")
def battlecard_status(job_id):
    """Poll this endpoint every few seconds to check job progress."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


def _run_battlecard(job_id: str, competitor: str):
    """Background worker — runs the full battle card pipeline and stores the result in _jobs."""
    try:
        today = date.today().strftime("%B %d, %Y")

        # ── Step 1: Web sentiment enrichment (G2 / TrustRadius) ───────────────
        _jobs[job_id]["step"] = "Step 1/3 — Fetching G2 & TrustRadius reviews…"
        print(f"[battlecard:{job_id}] Step 1: web search for {competitor!r}")
        sentiment_text = ""
        try:
            sentiment_msg = claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 1,
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search G2 and TrustRadius for user reviews of {competitor} specifically "
                        f"as an HR, HCM, or payroll software product. Only use reviews that relate to "
                        f"HR, payroll, workforce management, or people management use cases — ignore any "
                        f"reviews about other product lines this vendor may have.\n\n"
                        f"Extract and summarise:\n"
                        f"1. Top 3-5 things HR/payroll users love about {competitor} (recurring positive themes)\n"
                        f"2. Top 3-5 things HR/payroll users complain about (recurring negative themes)\n"
                        f"3. Segment patterns: Do enterprise HR teams say different things than mid-market HR teams?\n"
                        f"4. Recent trend: Has sentiment shifted in the last 12 months?\n\n"
                        f"Be specific to the HR/payroll context. Cite G2 or TrustRadius as the source for each point. "
                        f"Do not include sentiment from reviews about unrelated product areas."
                    ),
                }],
            )
            sentiment_text = "\n\n".join(
                block.text for block in sentiment_msg.content
                if hasattr(block, "text") and block.text.strip()
            )
            print(f"[battlecard:{job_id}] Step 1 done")
        except Exception as e:
            print(f"[battlecard:{job_id}] Step 1 FAILED: {e}")
            sentiment_text = f"[Web search unavailable: {e}]"

        # ── Step 2: Pull Pinecone context ──────────────────────────────────────
        _jobs[job_id]["step"] = "Step 2/3 — Searching knowledge base…"
        print(f"[battlecard:{job_id}] Step 2: Pinecone queries")

        def chunks_to_text(chunks):
            return "\n\n---\n\n".join(
                (m.metadata or {}).get("text", "") for m in chunks
            )

        competitor_chunks = vs.search(
            query_embedding=get_embedding(f"{competitor} competitor HR HCM payroll software"),
            top_k=4,
            filter={"doc_type": {"$eq": "competitor"}},
        )
        messaging_chunks = vs.search(
            query_embedding=get_embedding("Sage People value proposition differentiators why choose us"),
            top_k=3,
            filter={"doc_type": {"$eq": "messaging"}},
        )
        case_study_chunks = vs.search(
            query_embedding=get_embedding("customer win case study proof point success story outcome"),
            top_k=3,
            filter={"doc_type": {"$eq": "case_study"}},
        )

        all_chunks = list(competitor_chunks) + list(messaging_chunks) + list(case_study_chunks)
        source_files = sorted({
            (m.metadata or {}).get("file_name", "")
            for m in all_chunks
            if (m.metadata or {}).get("file_name")
        })
        competitor_context = chunks_to_text(competitor_chunks)
        messaging_context  = chunks_to_text(messaging_chunks)
        case_study_context = chunks_to_text(case_study_chunks)
        print(f"[battlecard:{job_id}] Step 2 done — {len(all_chunks)} chunks")

        # ── Step 3: Generate battle card ───────────────────────────────────────
        _jobs[job_id]["step"] = "Step 3/3 — Generating battle card with Claude…"
        print(f"[battlecard:{job_id}] Step 3: Claude generation")

        prompt = f"""You are a senior GTM strategist at Sage People, a cloud-native HR & HCM platform for 200–5,000 employee organisations.

Build a concise, honest battle card for competing against {competitor}. Sales reps scan this on their phone before a call — every bullet must be one sentence, punchy, and actionable.

=== EXTERNAL REVIEW SENTIMENT (G2 / TrustRadius) ===
{sentiment_text}

=== COMPETITOR INTELLIGENCE (Knowledge Base) ===
{competitor_context}

=== SAGE PEOPLE MESSAGING & DIFFERENTIATORS (Knowledge Base) ===
{messaging_context}

=== CUSTOMER PROOF POINTS (Knowledge Base) ===
{case_study_context}

Return ONLY a valid JSON object — no markdown, no extra text — matching this exact structure:
{{
  "competitor": "{competitor}",
  "segment": "Enterprise / Mid-market / Both",
  "generated_date": "{today}",
  "sources_used": {json.dumps(source_files)},
  "sentiment_summary": {{
    "source": "G2 and TrustRadius",
    "what_users_love": ["3 recurring positives — one sentence each, cite [G2] or [TrustRadius]"],
    "what_users_complain_about": ["3 recurring complaints — one sentence each, cite source"],
    "segment_patterns": "One sentence: how do enterprise vs mid-market reviewers differ?",
    "recent_trend": "One sentence on recent sentiment direction."
  }},
  "why_they_win": [
    "3 bullets — what {competitor} genuinely does well. One sentence each. End with [G2], [TrustRadius], or [Knowledge Base]."
  ],
  "where_we_lose": [
    "2 bullets — situations where {competitor} is the better fit. One sentence each. Be candid."
  ],
  "our_differentiators": [
    "4 bullets — why Sage People wins. One sentence each with a specific customer name or data point. End with [Knowledge Base]."
  ],
  "objections": [
    {{"objection": "Short verbatim objection reps hear", "response": "One-sentence counter with a real customer example or stat."}},
    {{"objection": "...", "response": "..."}},
    {{"objection": "...", "response": "..."}}
  ],
  "trap_questions": [
    "4 discovery questions starting with 'How do you...' or 'What happens when...'. Target {competitor}'s known weak spots from reviews."
  ],
  "proof_points": [
    {{"customer": "Name", "size": "Headcount / segment", "displaced": "What they replaced", "outcome": "One measurable result"}}
  ]
}}

RULES:
- One sentence per bullet. No exceptions.
- Be honest in why_they_win and where_we_lose — a one-sided card loses credibility.
- Only use claims grounded in the knowledge base or G2/TrustRadius. No invented facts.
- Direct, conversational language — internal use, not marketing copy.
- Always tag the source [Knowledge Base], [G2], or [TrustRadius] at the end of each bullet."""

        message = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        card = json.loads(raw)
        print(f"[battlecard:{job_id}] Done")
        _jobs[job_id] = {"status": "done", "result": card}

    except Exception as e:
        import traceback
        print(f"[battlecard:{job_id}] FAILED: {e}")
        _jobs[job_id] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    app.run(debug=True, port=5050)
