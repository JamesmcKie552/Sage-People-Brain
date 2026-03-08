#!/usr/bin/env python3
"""
competitor_sweep.py — Competitor intelligence sweep for Sage People Brain.

For each competitor, uses Claude with web search to gather recent intel and
writes structured data to static/competitor_intel.json.

Run manually:  python competitor_sweep.py
Flask calls:   from competitor_sweep import run_sweep
"""

import json
import os
import re
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

COMPETITORS = ["SAP SuccessFactors"]

OUTPUT_PATH = Path(__file__).parent / "static" / "competitor_intel.json"

# Sage People's known position — used in the prompt so Claude can
# highlight gaps and relevant comparisons.
SAGE_PROFILE = """
Sage People is a cloud HCM and payroll platform built natively on Salesforce.
- Strengths: Core HR, UK & Ireland payroll compliance, Salesforce-native (not just integrated),
  mid-market and enterprise (500–5,000 employees), fast implementation (3–6 months avg),
  strong customer support scores.
- Weaknesses vs large players: limited global/multi-country payroll, less depth in
  talent management and learning vs Workday/SAP, smaller brand recognition.
- Key differentiator: the only HCM platform natively built on Salesforce — not a connector.
"""


def _extract_text(response) -> str:
    """Pull text blocks from a Claude response (web search returns mixed blocks)."""
    parts = [block.text for block in response.content if hasattr(block, "text") and block.text.strip()]
    return "\n\n".join(parts)


def _extract_json(raw: str) -> str:
    """
    Best-effort extraction of a JSON object from raw text.
    Tries (in order): code fence, outermost {…} block, raw strip.
    """
    # 1. Code fence
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
    if match:
        return match.group(1).strip()
    # 2. Outermost { … } — handles "Here is the JSON: {...}"
    start = raw.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1].strip()
    return raw.strip()


def sweep_competitor(client: anthropic.Anthropic, name: str, step_cb=None) -> dict:
    """
    Research one competitor using Claude + web search, return structured dict.
    Uses claude-haiku for cost efficiency; max_uses=3 gives it enough search budget.
    """
    if step_cb:
        step_cb(f"Researching {name}…")

    prompt = f"""You are a competitive intelligence analyst for Sage People sales team.

Research {name} as an HCM/HR/payroll software product right now. Search for:
1. Their current market positioning — who they target, their headline value prop
2. Recent product launches, feature announcements, or pricing changes (last 6 months)
3. What they lead with in their website/marketing messaging (3–5 key angles)
4. Known weaknesses from G2, Capterra, or TrustRadius user reviews
5. Any notable news: leadership changes, layoffs, acquisitions, or major wins/losses

IMPORTANT: After searching, respond with ONLY a valid JSON object — no preamble, no explanation, no markdown fences. Start your response with {{ and end with }}.
{{
  "name": "{name}",
  "swept_at": "{date.today().isoformat()}",
  "overview": "2–3 sentence positioning summary: who they are, who they target, key differentiator",
  "sentiment_trend": "improving|stable|declining",
  "status_line": "One punchy line (max 12 words) on what's most notable about them RIGHT NOW",
  "recent_news": [
    {{
      "headline": "Short headline",
      "date": "YYYY-MM",
      "implication": "Why this matters for a Sage People rep in a competitive deal (1 sentence)"
    }}
  ],
  "feature_matrix": {{
    "salesforce_native": false,
    "payroll_included": true,
    "mid_market_focus": true,
    "global_payroll": true,
    "ai_features": "Brief description of their AI capabilities, or 'Limited'",
    "implementation_weeks": "e.g. 26–52 weeks",
    "pricing_model": "e.g. per-employee-per-month, enterprise contract"
  }},
  "messaging_angles": [
    "3–5 bullets: what they actually say about themselves in their marketing"
  ],
  "known_weaknesses": [
    "3–5 bullets: recurring complaints from real users (cite G2/Capterra where possible)"
  ],
  "gap_alerts": [
    {{
      "type": "product|pricing|messaging|people",
      "severity": "high|medium|low",
      "detail": "What changed and why it matters competitively (2 sentences max)",
      "talk_track": "How a Sage People rep should handle this on a call (1–2 sentences)"
    }}
  ]
}}

Rules:
- recent_news: 3–5 items, most recent first. Only real, verifiable items.
- gap_alerts: only genuine competitive threats from the last 6 months. Can be empty [].
- Be honest about where {name} is genuinely strong — a credible card loses nothing by being fair.
- All fields required. Use null for unknown values, never omit a key."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        messages=[{"role": "user", "content": prompt}],
    )

    raw = _extract_json(_extract_text(response))

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return a minimal error record rather than crashing the whole sweep
        data = {
            "name": name,
            "swept_at": date.today().isoformat(),
            "error": f"JSON parse failed. Raw response: {raw[:300]}",
        }

    # Ensure name + date are always set correctly
    data["name"] = name
    data["swept_at"] = date.today().isoformat()
    return data


def run_sweep(step_cb=None) -> list:
    """
    Run the full sweep for all competitors.
    step_cb(msg): optional callback for progress updates (used by Flask async job).
    Returns list of competitor dicts and writes to static/competitor_intel.json.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    for name in COMPETITORS:
        try:
            result = sweep_competitor(client, name, step_cb)
            results.append(result)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results.append({
                "name": name,
                "swept_at": date.today().isoformat(),
                "error": str(e),
            })

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    if step_cb:
        step_cb("Done — writing results…")

    return results


if __name__ == "__main__":
    print("Starting competitor sweep…")
    results = run_sweep(step_cb=print)
    successful = sum(1 for r in results if "error" not in r)
    print(f"\n✅ Sweep complete: {successful}/{len(results)} competitors OK → {OUTPUT_PATH}")
