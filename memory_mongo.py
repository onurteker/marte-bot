#!/usr/bin/env python3
"""
MarteMemory - Kalıcı Semantik Hafıza Sistemi
MongoDB Atlas (kalıcı) + Gemini text-embedding-004 ile semantik arama
+ Kullanıcı profili otomatik çıkarımı
"""

import os
import datetime
import numpy as np
import google.generativeai as genai

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

import json


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class MarteMemory:
    def __init__(self, gemini_api_key: str, mongodb_uri: str = None):
        genai.configure(api_key=gemini_api_key)

        # MongoDB bağlantısı
        uri = mongodb_uri or os.environ.get("MONGODB_URI", "")
        self.use_mongo = False

        if uri and MONGO_AVAILABLE:
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.server_info()
                db = client["marte_memory"]
                self.msg_col   = db["messages"]
                self.doc_col   = db["documents"]
                self.facts_col = db["user_facts"]
                self.use_mongo = True
                print("✅ MongoDB Atlas bağlantısı başarılı!")
            except Exception as e:
                print(f"⚠️ MongoDB bağlanamadı, JSON fallback: {e}")

        if not self.use_mongo:
            self.data_dir = "memory_data"
            os.makedirs(self.data_dir, exist_ok=True)
            self._msgs_path  = os.path.join(self.data_dir, "messages.json")
            self._docs_path  = os.path.join(self.data_dir, "documents.json")
            self._facts_path = os.path.join(self.data_dir, "user_facts.json")
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

    def _save_json(self, path, data):
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
        entry = {"type": "message", "role": role, "text": text, "user_id": user_id, "timestamp": self._ts(), "embedding": embedding}
        if self.use_mongo:
            self.msg_col.insert_one(entry)
            count = self.msg_col.count_documents({})
            if count > 1000:
                oldest = self.msg_col.find().sort("timestamp", 1).limit(count - 1000)
                ids = [d["_id"] for d in oldest]
                self.msg_col.delete_many({"_id": {"$in": ids}})
        else:
            self.messages.append(entry)
            if len(self.messages) > 1000:
                self.messages = self.messages[-1000:]
            self._save_json(self._msgs_path, self.messages)

    def add_document(self, filename: str, summary: str, mime_type: str = ""):
        embedding = self._embed(summary[:2000])
        entry = {"type": "document", "filename": filename, "summary": summary, "mime_type": mime_type, "timestamp": self._ts(), "embedding": embedding}
        if self.use_mongo:
            self.doc_col.insert_one(entry)
        else:
            self.documents.append(entry)
            self._save_json(self._docs_path, self.documents)

    def add_user_fact(self, fact: str):
        if self.use_mongo:
            existing = self.facts_col.find_one({"fact": {"$regex": f"^{fact}$", "$options": "i"}})
            if existing:
                return
            self.facts_col.insert_one({"fact": fact, "timestamp": self._ts(), "embedding": self._embed(fact)})
        else:
            for existing in self.user_facts:
                if existing.get("fact", "").lower().strip() == fact.lower().strip():
                    return
            self.user_facts.append({"fact": fact, "timestamp": self._ts(), "embedding": self._embed(fact)})
            self._save_json(self._facts_path, self.user_facts)

    def get_user_profile_text(self) -> str:
        if self.use_mongo:
            facts = [d["fact"] for d in self.facts_col.find().sort("timestamp", -1).limit(100)]
            facts.reverse()
        else:
            facts = [f["fact"] for f in self.user_facts[-100:]]
        if not facts:
            return ""
        return "Kullanici hakkinda bilinen bilgiler:\n" + "\n".join(f"- {f}" for f in facts)

    @property
    def user_facts_list(self):
        if self.use_mongo:
            return list(self.facts_col.find())
        return self.user_facts

    def auto_extract_facts(self, user_message: str, groq_client) -> list:
        if len(user_message) < 25:
            return []
        if self.use_mongo:
            existing_facts = [d["fact"] for d in self.facts_col.find().sort("timestamp", -1).limit(20)]
        else:
            existing_facts = [f["fact"] for f in self.user_facts[-20:]]
        existing_str = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "Henuz yok"
        prompt = f"""Asagidaki kullanici mesajindan, kullanici hakkinda KALICI ve YENI bilgiler cikar.\nOrnek: meslek, projeler, isim, tercihler, uzmanlik alanlari, hobiler, onemli kisiler.\n\nZaten bilinen bilgiler (tekrarlama yapma):\n{existing_str}\n\nKullanici mesaji: {user_message[:400]}\n\nSADECE yeni ve kalici bilgileri yaz. Her bilgi bir satir. Yoksa sadece "YOK" yaz.\nGecici seyler (bugunun havasi, anlik sorular) ekleme."""
        try:
            resp = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], max_tokens=150, temperature=0.1)
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

    def search(self, query: str, n: int = 5):
        qemb = self._embed_query(query)
        if qemb is None:
            return []
        all_entries = list(self.msg_col.find()) + list(self.doc_col.find()) if self.use_mongo else self.messages + self.documents
        results = []
        for entry in all_entries:
            emb = entry.get("embedding")
            if emb is None:
                continue
            score = _cosine(qemb, emb)
            results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:n]

    def get_context(self, query: str, n: int = 5) -> str:
        results = self.search(query, n=n)
        if not results:
            recent = list(self.msg_col.find().sort("timestamp", -1).limit(5)) if self.use_mongo else self.messages[-5:]
            if self.use_mongo:
                recent.reverse()
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
        if self.use_mongo:
            return {"messages": self.msg_col.count_documents({}), "documents": self.doc_col.count_documents({}), "user_facts": self.facts_col.count_documents({}), "storage": "MongoDB Atlas ✅ (kalici)"}
        return {"messages": len(self.messages), "documents": len(self.documents), "user_facts": len(self.user_facts), "storage": "JSON (gecici - restart'ta sifirlanir)"}
