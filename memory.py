#!/usr/bin/env python3
"""
MarteMemory - Kalici Semantik Hafiza Sistemi
Gemini text-embedding-004 ile semantik arama
"""

import os
import json
import datetime
import numpy as np
import google.generativeai as genai


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class MarteMemory:
    def __init__(self, gemini_api_key: str, data_dir: str = "memory_data"):
        genai.configure(api_key=gemini_api_key)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._msgs_path  = os.path.join(data_dir, "messages.json")
        self._docs_path  = os.path.join(data_dir, "documents.json")
        self._facts_path = os.path.join(data_dir, "user_facts.json")
        self.messages   = self._load(self._msgs_path)
        self.documents  = self._load(self._docs_path)
        self.user_facts = self._load(self._facts_path)

    def _load(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _embed(self, text: str):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            return result["embedding"]
        except Exception:
            return None

    def _embed_query(self, text: str):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="RETRIEVAL_QUERY",
            )
            return result["embedding"]
        except Exception:
            return None

    def _ts(self):
        return datetime.datetime.utcnow().isoformat()

    def add_message(self, role: str, text: str, user_id=None):
        embedding = self._embed(text[:2000])
        entry = {"type": "message", "role": role, "text": text, "user_id": user_id,
                 "timestamp": self._ts(), "embedding": embedding}
        self.messages.append(entry)
        if len(self.messages) > 500:
            self.messages = self.messages[-500:]
        self._save(self._msgs_path, self.messages)

    def add_document(self, filename: str, summary: str, mime_type: str = ""):
        embedding = self._embed(summary[:2000])
        entry = {"type": "document", "filename": filename, "summary": summary,
                 "mime_type": mime_type, "timestamp": self._ts(), "embedding": embedding}
        self.documents.append(entry)
        self._save(self._docs_path, self.documents)

    def search(self, query: str, n: int = 5):
        qemb = self._embed_query(query)
        if qemb is None:
            return []
        results = []
        for entry in self.messages + self.documents:
            emb = entry.get("embedding")
            if emb is None:
                continue
            results.append((_cosine(qemb, emb), entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:n]

    def get_context(self, query: str, n: int = 3) -> str:
        results = self.search(query, n=n)
        if not results:
            recent = self.messages[-5:]
            if not recent:
                return ""
            lines = ["Son konusmalar:"]
            for m in recent:
                lines.append(f"[{m['role']}]: {m['text'][:300]}")
            return "\n".join(lines)
        lines = ["Ilgili gecmis bagiam:"]
        for score, entry in results:
            if score < 0.3:
                continue
            if entry["type"] == "message":
                lines.append(f"[{entry['role']}]: {entry['text'][:300]}")
            else:
                lines.append(f"[Belge - {entry['filename']}]: {entry['summary'][:300]}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def stats(self):
        return {"messages": len(self.messages), "documents": len(self.documents),
                "user_facts": len(self.user_facts)}
