"""Keyword search over the knowledge base. The LLM chooses the keywords."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

KB_FILE = Path(__file__).resolve().parents[2] / "candidate" / "knowledge_base.json"
DOCS: list[dict] = json.loads(KB_FILE.read_text(encoding="utf-8"))


class SearchArgs(BaseModel):
    keywords: list[str] = Field(min_length=1, description="Keywords to look up, e.g. ['return', 'unopened']")


def search_knowledge_base(keywords: list[str]) -> str:
    """Search the company knowledge base by keywords. Returns matching documents with their IDs, best match first."""
    keywords = [k.lower().strip() for k in keywords if k.strip()]
    matches = []
    for doc in DOCS:
        text = f"{doc['title']} {doc['content']}".lower()
        hits = sum(k in text for k in keywords)
        if hits:
            matches.append((hits, doc))
    if not matches:
        return "No documents matched those keywords."
    matches.sort(key=lambda m: m[0], reverse=True)
    return "\n\n".join(f"[{d['id']}] {d['title']}\n{d['content']}" for _, d in matches)
