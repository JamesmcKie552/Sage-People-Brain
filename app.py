"""
Sage People Brain — Web App

Run locally:  python app.py
Deployed on:  Render (see render.yaml)
"""

import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

import vector_store as vs

load_dotenv()

app = Flask(__name__)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


if __name__ == "__main__":
    app.run(debug=True, port=5050)
