#!/usr/bin/env python3
"""
Marte - Kişisel AI Asistan (Telegram Bot)
Groq (Llama 3.3 70B) + Gemini Embeddings + Kalıcı Semantik Hafıza
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

from memory import MarteMemory
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

# Gemini sadece embeddings için
genai.configure(api_key=GEMINI_API_KEY)

# Kalıcı hafıza sistemi (embeddings için Gemini kullanır)
memory = MarteMemory(GEMINI_API_KEY)

CHAT_MODEL   = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-3.2-11b-vision-preview"

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
        f"Merhaba {user.first_name}! Ben Marte, senin kişisel AI asistanım.\n\n"
        "Bana her şeyi sorabilirsin. Resim dosyaları da gönderebilirsin.\n\n"
        "Komutlar:\n"
        "/yardim - Yardım menüsü\n"
        "/hafiza - Hafıza istatistikleri\n"
        "/hatirlat <sorgu> - Geçmişte arama yap"
    )
    await update.message.reply_text(text)


async def yardim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Marte Komutları:\n\n"
        "/start - Başlangıç mesajı\n"
        "/yardim - Bu menü\n"
        "/hafiza - Hafıza istatistikleri\n"
        "/hatirlat <sorgu> - Geçmiş konuşmalarda ara\n\n"
        "Desteklenen dosya türleri:\n"
        "• JPG, PNG, WEBP (görsel analizi)\n"
        "• TXT (metin çıkarımı)\n\n"
        "Her konuşmayı hafızama kaydediyorum."
    )
    await update.message.reply_text(text)


async def hafiza(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = memory.stats()
    text = (
        f"Hafıza İstatistikleri:\n\n"
        f"Mesajlar: {stats['messages']}\n"
        f"Belgeler: {stats['documents']}\n"
        f"Kullanıcı notları: {stats['user_facts']}"
    )
    await update.message.reply_text(text)


async def hatirlat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Kullanım: /hatirlat <arama terimi>")
        return
    sorgu = " ".join(context.args)
    await update.message.reply_text(f"'{sorgu}' için hafızamda arıyorum...")
    try:
        results = memory.search(sorgu, n=5)
        if not results:
            await update.message.reply_text("İlgili bir şey bulamadım.")
            return
        lines = [f"En yakın {len(results)} sonuç:\n"]
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
        await update.message.reply_text(f"Arama hatası: {str(e)[:200]}")


# ── Mesaj işleyici ─────────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user = update.effective_user

    ctx = memory.get_context(text)

    system_prompt = (
        "Sen Marte, Türkçe konuşan kişisel bir AI asistansın. "
        "Samimi, zeki ve yardımseversin."
    )

    if ctx:
        prompt = f"{ctx}\n\nŞu anki kullanıcı mesajı: {text}"
    else:
        prompt = text

    try:
        response_text = groq_chat(prompt, system=system_prompt)

        memory.add_message("user", text, user_id=user.id)
        memory.add_message("model", response_text)

        for i in range(0, len(response_text), 4000):
            await update.message.reply_text(response_text[i : i + 4000])

    except Exception as e:
        await update.message.reply_text(f"Hata: {str(e)[:200]}")


# ── Doküman işleyici ───────────────────────────────────────────────────────────
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    fname = doc.file_name or "dosya"
    mime = doc.mime_type or "application/octet-stream"
    caption = update.message.caption or "Bu dosyayı detaylıca analiz et ve özetle."

    await update.message.reply_text(f"Analiz ediyorum: {fname}...")

    try:
        tf = await context.bot.get_file(doc.file_id)
        fb = await tf.download_as_bytearray()

        if mime in ("text/plain",) or fname.endswith(".txt"):
            file_text = fb.decode("utf-8", errors="ignore")[:8000]
            prompt = f"{caption}\n\nDosya içeriği:\n{file_text}"
        else:
            file_text = f"[{fname} - {len(fb)} bytes, {mime}]"
            prompt = f"{caption}\n\nDosya: {file_text}\n(Not: Bu dosya türü için metin çıkarımı desteklenmiyor)"

        result_text = groq_chat(prompt)

        memory.add_document(fname, result_text[:1500], mime_type=mime)
        memory.add_message("user", f"[Dosya yüklendi: {fname}] {caption}")
        memory.add_message("model", result_text[:500])

        for i in range(0, len(result_text), 4000):
            await update.message.reply_text(result_text[i : i + 4000])

    except Exception as e:
        await update.message.reply_text(f"Hata: {str(e)[:200]}")


# ── Fotoğraf işleyici ──────────────────────────────────────────────────────────
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    caption = update.message.caption or "Bu görseli detaylıca analiz et."

    await update.message.reply_text("Görsel analiz ediliyor...")

    try:
        tf = await context.bot.get_file(photo.file_id)
        fb = await tf.download_as_bytearray()
        b64 = base64.b64encode(bytes(fb)).decode()

        result_text = groq_vision(b64, caption)

        memory.add_message("user", f"[Görsel gönderildi] {caption}")
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

    # Render.com free tier için ping sunucusunu başlat
    port = int(os.environ.get("PORT", 8080))
    start_ping_server(port)
    logger.info(f"Ping sunucusu port {port}'de başlatıldı.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("yardim", yardim))
    app.add_handler(CommandHandler("hafiza", hafiza))
    app.add_handler(CommandHandler("hatirlat", hatirlat))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Marte baslatiliyor...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
