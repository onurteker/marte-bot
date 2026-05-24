#!/usr/bin/env python3
"""
MarteMemory - Kalıcı Semantik Hafıza Sistemi
Gemini text-embedding-004 ile semantik arama
+ Kullanıcı profili otomatik çıkarımı
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

    # ── Yardımcılar ──────────────────────────────────────────────────────────

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

    # ── Mesaj Ekleme ──────────────────────────────────────────────────────────

    def add_message(self, role: str, text: str, user_id=None):
        embedding = self._embed(text[:2000])
        entry = {
            "type": "message",
            "role": role,
            "text": text,
            "user_id": user_id,
            "timestamp": self._ts(),
            "embedding": embedding,
        }
        self.messages.append(entry)
        # Son 1000 mesajı tut
        if len(self.messages) > 1000:
            self.messages = self.messages[-1000:]
        self._save(self._msgs_path, self.messages)

    # ── Belge Ekleme ──────────────────────────────────────────────────────────

    def add_document(self, filename: str, summary: str, mime_type: str = ""):
        embedding = self._embed(summary[:2000])
        entry = {
            "type": "document",
            "filename": filename,
            "summary": summary,
            "mime_type": mime_type,
            "timestamp": self._ts(),
            "embedding": embedding,
        }
        self.documents.append(entry)
        self._save(self._docs_path, self.documents)

    # ── Kullanıcı Profili ─────────────────────────────────────────────────────

    def add_user_fact(self, fact: str):
        """Kullanıcı hakkında kalıcı bir bilgi ekle"""
        for existing in self.user_facts:
            if existing.get("fact", "").lower().strip() == fact.lower().strip():
                return
        embedding = self._embed(fact)
        entry = {
            "fact": fact,
            "timestamp": self._ts(),
            "embedding": embedding,
        }
        self.user_facts.append(entry)
        self._save(self._facts_path, self.user_facts)

    def get_user_profile_text(self) -> str:
        """Sistem promptu icin kullanici profilini duzenli metne donustur"""
        if not self.user_facts:
            return ""
        facts = [f["fact"] for f in self.user_facts[-100:]]
        return "Kullanici hakkinda bilinen bilgiler:\n" + "\n".join(f"- {f}" for f in facts)

    def auto_extract_facts(self, user_message: str, groq_client) -> list:
        """
        Kullanici mesajindan kalici bilgileri otomatik cikar.
        """
        if len(user_message) < 25:
            return []

        existing_facts = [f["fact"] for f in self.user_facts[-20:]]
        existing_str = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "Henuz yok"

        prompt = f"""Asagidaki kullanici mesajindan, kullanici hakkinda KALICI ve YENI bilgiler cikar.
Ornek: meslek, projeler, isim, tercihler, uzmanlik alanlari, hobiler, onemli kisiler.

Zaten bilinen bilgiler (tekrarlama yapma):
{existing_str}

Kullanici mesaji: {user_message[:400]}

SADECE yeni ve kalici bilgileri yaz. Her bilgi bir satir. Yoksa sadece "YOK" yaz.
Gecici seyler (bugunun havasi, anlik sorular) ekleme."""

        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            )
            result = resp.choices[0].message.content.strip()
            if result.upper().startswith("YOK") or not result:
                return []
            facts = []
            for line in result.split("\n"):
                line = line.strip().lstrip("-*•").strip()
                if line and len(line) > 5 and "YOK" not in line.upper():
                    facts.append(line)
            return facts
        except Exception:
            return []

    # ── Semantik Arama ────────────────────────────────────────────────────────

    def search(self, query: str, n: int = 5):
        qemb = self._embed_query(query)
        if qemb is None:
            return []

        results = []
        all_entries = self.messages + self.documents

        for entry in all_entries:
            emb = entry.get("embedding")
            if emb is None:
                continue
            score = _cosine(qemb, emb)
            results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:n]

    # ── Bagiam Olusturma ─────────────────────────────────────────────────────

    def get_context(self, query: str, n: int = 5) -> str:
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

    # ── Istatistikler ─────────────────────────────────────────────────────────

    def stats(self):
        return {
            "messages": len(self.messages),
            "documents": len(self.documents),
            "user_facts": len(self.user_facts),
        }
