#!/usr/bin/env python3
"""
MarteMemory - KalÄ±cÄ± Semantik HafÄ±za Sistemi
MongoDB Atlas (kalÄ±cÄ±) + Gemini text-embedding-004 ile semantik arama
+ KullanÄ±cÄ± profili otomatik Ã§Ä±karÄ±mÄ±
"""

import os
import datetime
import numpy as np
from google import genai

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
        self._genai = genai.Client(api_key=gemini_api_key)

        # MongoDB baÄlantÄ±sÄ±
        uri = mongodb_uri or os.environ.get("MONGODB_URI", "")
        self.use_mongo = False

        if uri and MONGO_AVAILABLE:
            try:
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                client.server_info()  # BaÄlantÄ±yÄ± test et
                db = client["marte_memory"]
                self.msg_col   = db["messages"]
                self.doc_col   = db["documents"]
                self.facts_col = db["user_facts"]
                self.sys_col   = db["system_instructions"]
                self.use_mongo = True
                print("â MongoDB Atlas baÄlantÄ±sÄ± baÅarÄ±lÄ±!")
            except Exception as e:
                print(f"â ï¸ MongoDB baÄlanamadÄ±, JSON fallback: {e}")

        if not self.use_mongo:
            # JSON fallback (Render restart'ta sÄ±fÄ±rlanÄ±r ama Ã§alÄ±ÅÄ±r)
            self.data_dir = "memory_data"
            os.makedirs(self.data_dir, exist_ok=True)
            self._msgs_path  = os.path.join(self.data_dir, "messages.json")
            self._docs_path  = os.path.join(self.data_dir, "documents.json")
            self._facts_path = os.path.join(self.data_dir, "user_facts.json")
            self._sys_path   = os.path.join(self.data_dir, "system_instructions.json")
            self.messages        = self._load(self._msgs_path)
            self.documents       = self._load(self._docs_path)
            self.user_facts      = self._load(self._facts_path)
            self.sys_instructions = self._load(self._sys_path)

    # ââ JSON YardÄ±mcÄ±lar ââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

    # ââ Embedding âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _embed(self, text: str):
        try:
            result = self._genai.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            return list(result.embeddings[0].values)
        except Exception:
            return None

    def _embed_query(self, text: str):
        try:
            result = self._genai.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config={"task_type": "RETRIEVAL_QUERY"},
            )
            return list(result.embeddings[0].values)
        except Exception:
            return None

    def _ts(self):
        return datetime.datetime.utcnow().isoformat()

    # ââ Mesaj Ekleme ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
        if self.use_mongo:
            self.msg_col.insert_one(entry)
            # Son 1000 mesajÄ± tut
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

    # ââ Belge Ekleme ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
        if self.use_mongo:
            self.doc_col.insert_one(entry)
        else:
            self.documents.append(entry)
            self._save_json(self._docs_path, self.documents)

    # ââ KullanÄ±cÄ± Profili âââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def add_user_fact(self, fact: str):
        """KullanÄ±cÄ± hakkÄ±nda kalÄ±cÄ± bir bilgi ekle"""
        if self.use_mongo:
            existing = self.facts_col.find_one({"fact": {"$regex": f"^{fact}$", "$options": "i"}})
            if existing:
                return
            embedding = self._embed(fact)
            self.facts_col.insert_one({
                "fact": fact,
                "timestamp": self._ts(),
                "embedding": embedding,
            })
        else:
            for existing in self.user_facts:
                if existing.get("fact", "").lower().strip() == fact.lower().strip():
                    return
            embedding = self._embed(fact)
            self.user_facts.append({"fact": fact, "timestamp": self._ts(), "embedding": embedding})
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
                line = line.strip().lstrip("-*â¢").strip()
                if line and len(line) > 5 and "YOK" not in line.upper():
                    facts.append(line)
            return facts
        except Exception:
            return []

    # ââ Semantik Arama ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def search(self, query: str, n: int = 5):
        qemb = self._embed_query(query)
        if qemb is None:
            return []

        if self.use_mongo:
            all_entries = list(self.msg_col.find()) + list(self.doc_col.find())
        else:
            all_entries = self.messages + self.documents

        results = []
        for entry in all_entries:
            emb = entry.get("embedding")
            if emb is None:
                continue
            score = _cosine(qemb, emb)
            results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:n]

    # ââ BaÄlam OluÅturma ââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def get_context(self, query: str, n: int = 5) -> str:
        results = self.search(query, n=n)
        if not results:
            if self.use_mongo:
                recent = list(self.msg_col.find().sort("timestamp", -1).limit(5))
                recent.reverse()
            else:
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

    # ââ Sistem TalimatlarÄ± ââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def add_system_instruction(self, instruction: str) -> str:
        """KalÄ±cÄ± sistem talimatÄ± ekle, ID dÃ¶ndÃ¼r"""
        import hashlib
        inst_id = hashlib.md5(instruction.encode()).hexdigest()[:8]
        entry = {"id": inst_id, "instruction": instruction, "timestamp": self._ts()}
        if self.use_mongo:
            if not self.sys_col.find_one({"id": inst_id}):
                self.sys_col.insert_one(entry)
        else:
            if not any(s["id"] == inst_id for s in self.sys_instructions):
                self.sys_instructions.append(entry)
                self._save_json(self._sys_path, self.sys_instructions)
        return inst_id

    def get_system_instructions(self) -> list:
        """TÃ¼m sistem talimatlarÄ±nÄ± metin olarak dÃ¶ndÃ¼r"""
        if self.use_mongo:
            return [d["instruction"] for d in self.sys_col.find().sort("timestamp", 1)]
        return [s["instruction"] for s in self.sys_instructions]

    def list_system_instructions(self) -> list:
        """TÃ¼m sistem talimatlarÄ±nÄ± (id dahil) dÃ¶ndÃ¼r"""
        if self.use_mongo:
            return list(self.sys_col.find().sort("timestamp", 1))
        return self.sys_instructions

    def remove_system_instruction(self, inst_id: str) -> bool:
        """Belirtilen ID'li talimatÄ± sil"""
        if self.use_mongo:
            result = self.sys_col.delete_one({"id": inst_id})
            return result.deleted_count > 0
        else:
            before = len(self.sys_instructions)
            self.sys_instructions = [s for s in self.sys_instructions if s["id"] != inst_id]
            if len(self.sys_instructions) < before:
                self._save_json(self._sys_path, self.sys_instructions)
                return True
            return False

    def auto_update_behavior(self, groq_client) -> list:
        """Konusma gecmisinden davranis tercihlerini cikar, sistem talimatlarini guncelle"""
        # Son 20 mesaji al
        if self.use_mongo:
            recent = list(self.msg_col.find().sort("timestamp", -1).limit(20))
            recent.reverse()
        else:
            recent = self.messages[-20:]

        if len(recent) < 6:
            return []

        msgs_text = ""
        for m in recent:
            role = m.get("role", "")
            text = m.get("text", "")[:300]
            msgs_text += f"[{role}]: {text}\n"

        existing = self.get_system_instructions()
        existing_str = "\n".join(f"- {e}" for e in existing) if existing else "Henuz yok"

        prompt = f"""Asagidaki konusma gecmisini analiz et. Kullanicinin tercihlerini cikar.
Bak:
- Cevap uzunlugu (kisa mi uzun mu istedi, sikayeti var mi?)
- Ton (resmi mi samimi mi?)
- Tekrar eden konular veya odak alanlari
- Begenmemesini ima ettigi seyler

Mevcut aktif sistem talimatlari (bunlari tekrarlama):
{existing_str}

Konusma gecmisi:
{msgs_text}

SADECE yeni ve farkli, somut talimatlar yaz. Her biri bir satir.
Belirsiz veya genel seyler yazma. Max 2 talimat.
Yeni bir sey yoksa sadece "YOK" yaz."""

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
            new_instructions = []
            for line in result.split("\n"):
                line = line.strip().lstrip("-*â¢0123456789.").strip()
                if line and len(line) > 10 and "YOK" not in line.upper():
                    new_instructions.append(line)
            return new_instructions
        except Exception:
            return []

    # ââ Ä°statistikler âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def stats(self):
        if self.use_mongo:
            return {
                "messages": self.msg_col.count_documents({}),
                "documents": self.doc_col.count_documents({}),
                "user_facts": self.facts_col.count_documents({}),
                "storage": "MongoDB Atlas â (kalici)"
            }
        return {
            "messages": len(self.messages),
            "documents": len(self.documents),
            "user_facts": len(self.user_facts),
            "storage": "JSON (gecici - restart'ta sifirlanir)"
        }
