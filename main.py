#!/usr/bin/env python3
"""
Marte - Kisisel AI Asistan (Telegram Bot)
Groq (Llama 3.3 70B) + Gemini Embeddings + Kalici Semantik Hafiza
+ Otomatik kullanici profili + Web arama + PDF destegi
"""

import os
import json
import logging
import datetime
import asyncio
import base64

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from groq import Groq
import google.generativeai as genai

from memory_mongo import MarteMemory
from web_search import web_search
from render_keep_alive import start_ping_server

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
MONGODB_URI = os.environ.get("MONGODB_URI", "")
memory = MarteMemory(GEMINI_API_KEY, mongodb_uri=MONGODB_URI)

CHAT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"


def build_system_prompt() -> str:
    base = (
        "Sen Marte, Turkce konusan kisisel bir AI asistansin. "
        "Samimi, zeki ve yardimseversn. "
        "Kullanicinin kisisel asistanisin; onun projelerini, tercihlerini ve uzmanlik alanini biliyorsun.\n"
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


# ── Komutlar ──────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"Merhaba {user.first_name}! Ben Marte, senin kisisel AI asistaninim.\n\n"
        "Bana her seyi sorabilirsin. Resim, PDF, TXT dosyalari da gonderebilirsin.\n\n"
        "Komutlar:\n"
        "/yardim - Yardim menusu\n"
        "/hafiza - Hafiza istatistikleri\n"
        "/profil - Hakkimda bildiklerimi goster\n"
        "/ogret <bilgi> - Bana bir sey ogret\n"
        "/ara <sorgu> - Web'de ara\n"
        "/hatirlat <sorgu> - Gecmiste arama yap\n"
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
        "/ara <sorgu> - Web'de arama yap\n"
        "/hatirlat <sorgu> - Gecmis konusmalarda ara\n"
        "/sistem <talimat> - Kalici davranis talimati ekle/listele\n"
        "/sistem_sil <id> - Talimat sil\n\n"
        "Desteklenen dosya turleri:\n"
        "* JPG, PNG, WEBP (gorsel analizi)\n"
        "* PDF (metin cikarimi)\n"
        "* TXT (metin analizi)\n\n"
        "Her konusmani hafizama kaydediyorum ve seni zamanla daha iyi taniyorum."
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
            f"Asagidaki web arama sonuclarini Turkce olarak ozet ve kullanicinin sorusunu cevapla:\n\n"
            f"{results}\n\n"
            f"Kullanici sorusu: {sorgu}"
        )
        summary = groq_chat(summary_prompt, system=build_system_prompt())
        await update.message.reply_text(summary[:4000])
    except Exception as e:
        await update.message.reply_text(f"Arama hatasi: {str(e)[:200]}")


async def hatirlat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanim: /hatirlat <arama terimi>")
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
                "/sistem Sen bir kuantum fizigi uzmanisin\n\n"
                "Talimat listesi icin: /sistem (argumansiz)\n"
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


# ── Mesaj isleyici ─────────────────────────────────────────────────────────────
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
        response_text = groq_chat(prompt, system=system_prompt)

        memory.add_message("user", text, user_id=user.id)
        memory.add_message("model", response_text)

        # Otomatik kullanici profili cikarimi
        try:
            new_facts = memory.auto_extract_facts(text, groq_client)
            for fact in new_facts:
                memory.add_user_fact(fact)
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


# ── Dokuman isleyici ───────────────────────────────────────────────────────────
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    fname = doc.file_name or "dosya"
    mime = doc.mime_type or "application/octet-stream"
    caption = update.message.caption or "Bu dosyayi detaylica analiz et ve ozetle."

    await update.message.reply_text(f"Analiz ediyorum: {fname}...")

    try:
        tf = await context.bot.get_file(doc.file_id)
        fb = await tf.download_as_bytearray()

        if mime in ("text/plain",) or fname.endswith(".txt"):
            file_text = fb.decode("utf-8", errors="ignore")[:8000]
            prompt = f"{caption}\n\nDosya icerigi:\n{file_text}"

        elif mime == "application/pdf" or fname.lower().endswith(".pdf"):
            try:
                import io
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(bytes(fb)))
                pages_text = []
                for page in reader.pages[:20]:
                    pages_text.append(page.extract_text() or "")
                file_text = "\n".join(pages_text)[:8000]
                if file_text.strip():
                    prompt = f"{caption}\n\nPDF icerigi ({len(reader.pages)} sayfa):\n{file_text}"
                else:
                    prompt = f"{caption}\n\nPDF: {fname} - Metin cikarildi ancak icerik bos (taranmis gorsel PDF olabilir)"
            except Exception as pe:
                prompt = f"{caption}\n\nPDF: {fname} ({len(fb)} bytes) - Okuma hatasi: {str(pe)[:100]}"

        else:
            file_text = f"[{fname} - {len(fb)} bytes, {mime}]"
            prompt = f"{caption}\n\nDosya: {file_text}\n(Bu dosya turu icin metin cikarimi desteklenmiyor)"

        result_text = groq_chat(prompt, system=build_system_prompt())

        memory.add_document(fname, result_text[:1500], mime_type=mime)
        memory.add_message("user", f"[Dosya yuklendi: {fname}] {caption}")
        memory.add_message("model", result_text[:500])

        for i in range(0, len(result_text), 4000):
            await update.message.reply_text(result_text[i : i + 4000])

    except Exception as e:
        await update.message.reply_text(f"Hata: {str(e)[:200]}")


# ── Fotograf isleyici ──────────────────────────────────────────────────────────
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


# ── Ana fonksiyon ──────────────────────────────────────────────────────────────
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
    app.add_handler(CommandHandler("hatirlat", hatirlat))
    app.add_handler(CommandHandler("sistem", sistem))
    app.add_handler(CommandHandler("sistem_sil", sistem_sil))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Marte baslatiliyor...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
