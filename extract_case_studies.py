#!/usr/bin/env python3
"""
Extract structured Challenge / Solution / Results content from each
Sage People case study PDF and save to static/case_studies.json.

Run once (or whenever new case studies are added):
    python extract_case_studies.py
"""

import json
import re
from pathlib import Path

import fitz
import anthropic
from dotenv import load_dotenv

load_dotenv()

CASE_STUDIES_DIR = Path("Case Studies")
OUTPUT_FILE = Path("static/case_studies.json")

client = anthropic.Anthropic()


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    raw = "\n".join(pages)
    # Basic cleanup
    raw = re.sub(r"-\n(\w)", r"\1", raw)
    raw = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
    raw = re.sub(r" {2,}", " ", raw)
    return raw.strip()


def parse_filename(fname: str) -> dict:
    stem = fname.replace(".pdf", "")
    parts = [p.strip() for p in stem.split("\u2013")]  # em dash
    # Also try hyphen-dash for older naming style
    if len(parts) < 2:
        parts = [p.strip() for p in stem.split(" - ")]
    return {
        "company":  parts[0] if len(parts) > 0 else stem,
        "industry": parts[1] if len(parts) > 1 else "",
        "product":  parts[2] if len(parts) > 2 else "",
        "tagline":  parts[3] if len(parts) > 3 else "",
    }


def extract_story(pdf_path: Path) -> dict:
    text = extract_text(pdf_path)
    parsed = parse_filename(pdf_path.name)

    prompt = f"""You are reading a Sage People customer case study PDF.
Extract the following from the text and return ONLY a valid JSON object — no markdown, no extra text.

PDF text:
{text[:6000]}

Return this exact JSON structure:
{{
  "challenge": "2-3 sentences describing the business challenge or problem the customer faced before Sage People. Be specific — include pain points, manual processes, scale issues, etc.",
  "solution": "2-3 sentences describing how Sage People solved the problem. Focus on what was implemented and how.",
  "results": "2-3 sentences of measurable outcomes and benefits achieved. Include specific metrics, time savings, or business impact where mentioned.",
  "quote": {{
    "text": "The single most compelling direct quote from the case study, or empty string if none found",
    "name": "Full name of the person quoted, or empty string",
    "title": "Job title of the person quoted, or empty string",
    "company": "{parsed['company']}"
  }},
  "company_size": "Headcount or employee range if mentioned, e.g. '2,000 employees' or empty string",
  "country": "Primary country or region of the customer if mentioned, or empty string"
}}"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    story = json.loads(raw)
    story.update({
        "file_name": pdf_path.name,
        "company":   parsed["company"],
        "industry":  parsed["industry"],
        "product":   parsed["product"],
        "tagline":   parsed["tagline"],
    })
    return story


def main():
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    pdfs = sorted(CASE_STUDIES_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} case study PDFs\n")

    stories = []
    for pdf in pdfs:
        print(f"  Processing: {pdf.name[:60]}...")
        try:
            story = extract_story(pdf)
            stories.append(story)
            print(f"    ✅  {story['company']} — {len(story['challenge'])} char challenge")
        except Exception as e:
            print(f"    ❌  FAILED: {e}")

    OUTPUT_FILE.write_text(json.dumps(stories, indent=2, ensure_ascii=False))
    print(f"\n✅  Saved {len(stories)} stories to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
