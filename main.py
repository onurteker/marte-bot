#!/usr/bin/env python3
"""
Marte - Kisisel AI Asistan (Telegram Bot) v2
Groq (Llama 3.3 70B) + Gemini Embeddings + Kalici Semantik Hafiza
+ Otomatik Tool Use + 100 Sayfa PDF + Adim Adim Dusunce
"""

import os
import json
import logging
import asyncio
import base64

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters,
)
from groq import Groq

from memory_mongo import MarteMemory
from web_search import web_search
from render_keep_alive import start_ping_server

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)
MONGODB_URI = os.environ.get("MONGODB_URI", "")
memory = MarteMemory(GEMINI_API_KEY, mongodb_uri=MONGODB_URI)

CHAT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"

# ── Tool Definitions ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Web'de arama yap. Guncel haberler, bilmedigin konular, "
                "fiyatlar, olaylar veya herhangi bir guncel bilgi icin kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama sorgusu"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Gecmis konusmalarda ve yuklenen belgelerde ara. "
                "Kullanicinin daha once soylediklerini, paylastigini veya "
                "yuklenen dosyalari bulmak icin kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama sorgusu"}
                },
                "required": ["query"]
            }
        }
    }
]

# ── System Prompt ────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    base = (
        "Sen Marte, Turkce konusan kisisel bir AI asistansin. "
        "Samimi, zeki ve yardımseversin. "
        "Kullanicinin kisisel asistanisin; onun projelerini, tercihlerini ve uzmanlik alanini biliyorsun.\n"
        "Karmasik sorularda once adim adim dusun, sonra cevap ver. "
        "Guncel bilgi gerektiginde web_search, gecmis konusmalar gerektiginde search_memory kullan.\n"
    )
    profile = memory.get_user_profile_text()
    if profile:
        base += f"\n{profile}\n"
    instructions = memory.get_system_instructions()
    if instructions:
        base += "\nKalici sistem talimatlari (bunlara her zaman uy):\n"
        for inst in instructions:
            base += f"- {inst}\n"
    return base

# ── Tool Execution ───────────────────────────────────────────────────────────
def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "web_search":
            results = web_search(args.get("query", ""), max_results=4)
            return str(results)
        elif name == "search_memory":
            results = memory.search(args.get("query", ""), n=5)
            if not results:
                return "Hafizada ilgili bir sey bulunamadi."
            lines = []
            for score, entry in results:
                ts = entry.get("timestamp", "")[:10]
                if entry["type"] == "message":
                    lines.append(f"[{ts}] [{entry['role']}]: {entry['text'][:300]}")
                else:
                    lines.append(
                        f"[{ts}] [Dosya: {entry.get('filename','')}]: {entry.get('summary','')[:300]}"
                    )
            return "\n\n".join(lines)
        return "Bilinmeyen arac."
    except Exception as e:
        return f"Arac hatasi: {str(e)[:100]}"

# ── Groq Chat (basit, aracsiz) ────────────────────────────────────────────────
def groq_chat(prompt: str, system: str = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = groq_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=2048,
    )
    return resp.choices[0].message.content

# ── Groq Chat with Tools (otomatik arac secimi) ───────────────────────────────────
def groq_chat_with_tools(prompt: str, system: str = None) -> str:
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for _ in range(2):  # maksimum 2 tur arac cagrisi
            try:
                resp = groq_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=2048,
                )
            except Exception as api_err:
                logger.warning(f"Tool use API hatasi, aracsiz fallback: {api_err}")
                return groq_chat(prompt, system=system)

            msg = resp.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            # Asistan mesajini tool_calls ile ekle
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })

            # Araclari calistir ve sonuclari ekle
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                result = execute_tool(tc.function.name, args)
                logger.info(f"Tool cagrildi: {tc.function.name} -> {str(result)[:80]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:3000]
                })

        # Maksimum tur asildi: araclar olmadan son cevap
        try:
            resp = groq_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return groq_chat(prompt, system=system)

    except Exception as e:
        logger.error(f"groq_chat_with_tools genel hata: {e}")
        return groq_chat(prompt, system=system)

# ── Groq Vision ──────────────────────────────────────────────────────────────
def groq_vision(image_b64: str, prompt: str) -> str:
    resp = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=1024,
    )
    return resp.choices[0].message.content

# ── PDF Okuyucu ──────────────────────────────────────────────────────────────
def read_pdf_text(fb: bytes, max_pages: int = 100) -> tuple:
    """(tam_metin, toplam_sayfa) dondurur. PyMuPDF oncelikli, PyPDF2 yedek."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=fb, filetype="pdf")
        total = len(doc)
        pages_to_read = min(total, max_pages)
        chunks = []
        for i in range(pages_to_read):
            text = doc[i].get_text()
            if text.strip():
                chunks.append(f"[Sayfa {i+1}]\n{text}")
        doc.close()
        return "\n\n".join(chunks), total
    except ImportError:
        import io
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(bytes(fb)))
        total = len(reader.pages)
        pages_to_read = min(total, max_pages)
        chunks = []
        for i, page in enumerate(reader.pages[:pages_to_read]):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(f"[Sayfa {i+1}]\n{text}")
        return "\n\n".join(chunks), total

def process_pdf(fb: bytes, fname: str, caption: str, system: str) -> str:
    """PDF'i oku, gerekirse chunk'la ve ozetle."""
    full_text, total = read_pdf_text(fb)
    MAX_CHARS = 12000

    if not full_text.strip():
        return f"PDF: {fname} - Metin cikarildi ancak icerik bos (taranmis gorsel PDF olabilir)"

    if len(full_text) <= MAX_CHARS:
        prompt = f"{caption}\n\nPDF icerigi ({total} sayfa):\n{full_text}"
        return groq_chat_with_tools(prompt, system=system)

    # Uzun PDF: chunk'la, ozetle, birlestir
    chunk_size = 8000
    raw_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    max_chunks = min(len(raw_chunks), 12)

    logger.info(f"PDF {fname}: {total} sayfa, {len(raw_chunks)} chunk, {max_chunks} isleniyor")

    chunk_summaries = []
    for idx in range(max_chunks):
        chunk_prompt = (
            f"Bu PDF'nin {idx+1}/{max_chunks}. bolumu (dosya: {fname}, toplam {total} sayfa). "
            f"Onemli bilgileri koru, kisaca Turkce ozetle:\n\n{raw_chunks[idx]}"
        )
        summary = groq_chat(chunk_prompt, system="Metni kisaca ozetle, onemli bilgileri koru.")
        chunk_summaries.append(f"Bolum {idx+1}: {summary}")

    combined = "\n\n".join(chunk_summaries)
    final_prompt = (
        f"{caption}\n\nPDF - {fname} ({total} sayfa, {max_chunks} bolum analiz edildi):\n\n{combined}"
    )
    return groq_chat_with_tools(final_prompt, system=system)

# ── Komutlar ─────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"Merhaba {user.first_name}! Ben Marte, senin kisisel AI asistaninim.\n\n"
        "Bana her seyi sorabilirsin. Resim, PDF (100 sayfaya kadar), TXT dosyalari da gonderebilirsin.\n"
        "Gerektiginde otomatik web araması yapar, gecmis konusmalarinda sorgularim.\n\n"
        "Komutlar:\n"
        "/yardim - Yardim menusu\n"
        "/hafiza - Hafiza istatistikleri\n"
        "/profil - Hakkimda bildiklerimi goster\n"
        "/ogret <bilgi> - Bana kalici bir bilgi ogret\n"
        "/ara <sorgu> - Web'de ara\n"
        "/hatirla <sorgu> - Gecmiste arama yap\n"
        "/sistem <talimat> - Kalici davranis talimati ekle\n"
        "/sistem_sil <id> - Talimat sil"
    )
    await update.message.reply_text(text)

async def yardim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Marte Komutlari:\n\n"
        "/start - Baslangic mesaji\n"
        "/yardim - Bu menu\n"
        "/hafiza - Hafiza istatistikleri\n"
        "/profil - Hakkimda bildiklerimi goster\n"
        "/ogret <bilgi> - Bana kalici bir bilgi ogret\n"
        "/ara <sorgu> - Web'de ara\n"
        "/hatirla <sorgu> - Gecmis konusmalarda ara\n"
        "/sistem <talimat> - Kalici davranis talimati ekle/listele\n"
        "/sistem_sil <id> - Talimat sil\n\n"
        "Desteklenen dosya turleri:\n"
        "* JPG, PNG, WEBP (gorsel analizi)\n"
        "* PDF (100 sayfaya kadar metin cikarimi)\n"
        "* TXT (metin analizi)\n\n"
        "Her konusmani hafizama kaydediyorum ve seni zamanla daha iyi taniyorum.\n"
        "Guncel bilgi gerektiginde otomatik web araması yaparim."
    )
    await update.message.reply_text(text)

async def hafiza(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = memory.stats()
    text = (
        f"Hafiza Istatistikleri:\n\n"
        f"Depolama: {stats.get('storage', 'JSON')}\n"
        f"Mesajlar: {stats['messages']}\n"
        f"Belgeler: {stats['documents']}\n"
        f"Hakkinda bildigim bilgiler: {stats['user_facts']}"
    )
    await update.message.reply_text(text)

async def profil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    facts_list = memory.user_facts_list
    if not facts_list:
        await update.message.reply_text(
            "Henuz hakkinda kaydedilmis bilgi yok.\n"
            "/ogret komutuyla bana bilgi ogretebilirsin."
        )
        return
    facts = [f["fact"] for f in facts_list]
    text = f"Hakkinda bildiklerim ({len(facts)} bilgi):\n\n"
    for i, fact in enumerate(facts, 1):
        text += f"{i}. {fact}\n"
    await update.message.reply_text(text[:4000])

async def ogret(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Kullanim: /ogret <ogretmek istedigin bilgi>\n"
            "Ornek: /ogret Ben fizik arastirmacisiyim"
        )
        return
    fact = " ".join(context.args)
    memory.add_user_fact(fact)
    await update.message.reply_text(f"Kaydettim! \"{fact}\" - Artik bunu hep bilecegim.")

async def ara(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanim: /ara <arama terimi>")
        return
    sorgu = " ".join(context.args)
    await update.message.reply_text(f"'{sorgu}' icin web'de ariyorum...")
    try:
        results = web_search(sorgu, max_results=4)
        summary_prompt = (
            f"Asagidaki web arama sonuclarini Turkce olarak ozetle ve kullanicinin sorusunu cevapla:\n\n"
            f"{results}\n\n"
            f"Kullanici sorusu: {sorgu}"
        )
        summary = groq_chat(summary_prompt, system=build_system_prompt())
        await update.message.reply_text(summary[:4000])
    except Exception as e:
        await update.message.reply_text(f"Arama hatasi: {str(e)[:200]}")

async def hatirla(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanim: /hatirla <arama terimi>")
        return
    sorgu = " ".join(context.args)
    await update.message.reply_text(f"'{sorgu}' icin hafizamda ariyorum...")
    try:
        results = memory.search(sorgu, n=5)
        if not results:
            await update.message.reply_text("Ilgili bir sey bulamadim.")
            return
        lines = [f"En yakin {len(results)} sonuc:\n"]
        for score, entry in results:
            ts = entry.get("timestamp", "")[:10]
            if entry["type"] == "message":
                lines.append(
                    f"[{ts}] [{entry['role']}] (uyum: {score:.2f})\n{entry['text'][:200]}"
                )
            else:
                lines.append(
                    f"[{ts}] [Dosya: {entry['filename']}] (uyum: {score:.2f})\n{entry['summary'][:200]}"
                )
        await update.message.reply_text("\n\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Arama hatasi: {str(e)[:200]}")

async def sistem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        mevcut = memory.list_system_instructions()
        if mevcut:
            lines = [f"Aktif sistem talimatlari ({len(mevcut)}):\n"]
            for entry in mevcut:
                lines.append(f"[{entry['id']}] {entry['instruction']}")
            lines.append("\nSilmek icin: /sistem_sil <id>")
            lines.append("Eklemek icin: /sistem <talimat>")
            await update.message.reply_text("\n".join(lines))
        else:
            await update.message.reply_text(
                "Kullanim: /sistem <kalici talimat>\n\n"
                "Ornekler:\n"
                "/sistem Her zaman Turkce cevap ver\n"
                "/sistem Cevaplarini kisa ve oz tut\n"
                "/sistem Sen bir kuantum fizigi uzmanlisin\n\n"
                "Talimat listesi icin: /sistem (argumanssiz)\n"
                "Silmek icin: /sistem_sil <id>"
            )
        return
    instruction = " ".join(context.args)
    inst_id = memory.add_system_instruction(instruction)
    await update.message.reply_text(
        f"Kalici talimat eklendi (ID: {inst_id}):\n\"{instruction}\"\n\n"
        "Bu talimat bundan sonraki tum konusmalarda gecerli olacak."
    )

async def sistem_sil(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanim: /sistem_sil <id>\nMevcut talimatlar icin: /sistem")
        return
    inst_id = context.args[0]
    success = memory.remove_system_instruction(inst_id)
    if success:
        await update.message.reply_text(f"Talimat silindi: {inst_id}")
    else:
        await update.message.reply_text(f"Talimat bulunamadi: {inst_id}\nMevcut talimatlar icin: /sistem")

# ── Mesaj Isleyici ───────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user = update.effective_user

    ctx = memory.get_context(text)
    system_prompt = build_system_prompt()

    if ctx:
        prompt = f"{ctx}\n\nSu anki kullanici mesaji: {text}"
    else:
        prompt = text

    try:
        response_text = groq_chat_with_tools(prompt, system=system_prompt)

        memory.add_message("user", text, user_id=user.id)
        memory.add_message("model", response_text)

        # Otomatik kullanici profili cikarimi
        try:
            new_facts = memory.auto_extract_facts(text, groq_client)
            for fact in new_facts:
                memory.add_userfact(fact)
                logger.info(f"Yeni kullanici bilgisi: {fact}")
        except Exception:
            pass

        # Her 10 mesajda bir otomatik davranis ogrenimi
        try:
            msg_count = memory.stats()["messages"]
            if msg_count % 10 == 0:
                new_behaviors = memory.auto_update_behavior(groq_client)
                for behavior in new_behaviors:
                    memory.add_system_instruction(behavior)
                    logger.info(f"Otomatik davranis talimati eklendi: {behavior}")
        except Exception:
            pass

        for i in range(0, len(response_text), 4000):
            await update.message.reply_text(response_text[i : i + 4000])

    except Exception as e:
        await update.message.reply_text(f"Hata: {str(e)[:200]}")

# ── Dokuman Isleyici ─────────────────────────────────────────────────────────
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    fname = doc.file_name or "dosya"
    mime = doc.mime_type or "application/octet-stream"
    caption = update.message.caption or "Bu dosyayi detaylica analiz et ve ozetle."

    await update.message.reply_text(f"Analiz ediyorum: {fname}...")

    try:
        tf = await context.bot.get_file(doc.file_id)
        fb = await tf.download_as_bytearray()

        system = build_system_prompt()

        if mime in ("text/plain",) or fname.endswith(".txt"):
            file_text = fb.decode("utf-8", errors="ignore")[:8000]
            prompt = f"{caption}\n\nDosya icerigi:\n{file_text}"
            result_text = groq_chat_with_tools(prompt, system=system)

        elif mime == "application/pdf" or fname.lower().endswith(".pdf"):
            result_text = process_pdf(bytes(fb), fname, caption, system)

        else:
            file_text = f"[{fname} - {len(fb)} bytes, {mime}]"
            prompt = f"{caption}\n\nDosya: {file_text}\n(Bu dosya turu icin metin cikarimi desteklenmiyor)"
            result_text = groq_chat_with_tools(prompt, system=system)

        memory.add_document(fname, result_text[:1500], mime_type=mime)
        memory.add_message("user", f"[Dosya yuklendi: {fname}] {caption}")
        memory.add_message("model", result_text[:500])

        for i in range(0, len(result_text), 4000):
            await update.message.reply_text(result_text[i : i + 4000])

    except Exception as e:
        await update.message.reply_text(f"Hata: {str(e)[:200]}")

# ── Fotograf Isleyici ────────────────────────────────────────────────────────
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    caption = update.message.caption or "Bu gorseli detaylica analiz et."

    await update.message.reply_text("Gorsel analiz ediliyor...")

    try:
        tf = await context.bot.get_file(photo.file_id)
        fb = await tf.download_as_bytearray()
        b64 = base64.b64encode(bytes(fb)).decode()

        result_text = groq_vision(b64, caption)

        memory.add_message("user", f"[Gorsel gonderildi] {caption}")
        memory.add_message("model", result_text[:500])

        for i in range(0, len(result_text), 4000):
            await update.message.reply_text(result_text[i : i + 4000])

    except Exception as e:
        await update.message.reply_text(f"Hata: {str(e)[:200]}")

# ── Ana Fonksiyon ────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable eksik!")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable eksik!")

    port = int(os.environ.get("PORT", 8080))
    start_ping_server(port)
    logger.info(f"Ping sunucusu port {port}'de baslatildi.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("yardim", yardim))
    app.add_handler(CommandHandler("hafiza", hafiza))
    app.add_handler(CommandHandler("profil", profil))
    app.add_handler(CommandHandler("ogret", ogret))
    app.add_handler(CommandHandler("ara", ara))
    app.add_handler(CommandHandler("hatirla", hatirla))
    app.add_handler(CommandHandler("sistem", sistem))
    app.add_handler(CommandHandler("sistem_sil", sistem_sil))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Marte v2 baslatiliyor...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
