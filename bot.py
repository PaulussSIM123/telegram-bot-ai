import base64
import io
import logging
import os
from collections import defaultdict, deque

import httpx
from dotenv import load_dotenv
from PIL import Image
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("Error: TELEGRAM_BOT_TOKEN belum diset!")

if not OPENROUTER_API_KEY:
    raise ValueError("Error: OPENROUTER_API_KEY belum diset!")

ALLOWED_USERS = set(
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USERS", "").split(",")
    if uid.strip().isdigit()
)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Kamu adalah asisten AI yang ramah dan siap membantu menganalisis teks dan gambar.",
)

history = defaultdict(lambda: deque(maxlen=10))

# DAFTAR MODEL GRATIS AKTIF DI OPENROUTER (Vision & Text)
# Mengabaikan os.getenv agar tidak terpengaruh env variable lama di Railway
FREE_MODELS = [
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "deepseek/deepseek-r1:free",
    "mistralai/mistral-7b-instruct:free",
]


def compress_image(image_bytes: bytes, max_size: int = 1024) -> str:
    """Mengompres gambar dan mengembalikan string base64 JPEG."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((max_size, max_size))

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def ask_ai(messages: list[dict]) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://telegram.org",
        "X-Title": "Telegram AI Bot",
    }
    last_error = None
    async with httpx.AsyncClient(timeout=90) as client:
        for model in FREE_MODELS:
            try:
                log.info(f"Mencoba request ke OpenRouter model: {model}")
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": messages},
                )
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                if text and text.strip():
                    return text.strip()
            except Exception as e:
                last_error = e
                status_code = getattr(
                    getattr(e, "response", None), "status_code", None
                )
                response_text = getattr(
                    getattr(e, "response", None), "text", ""
                )
                log.warning(
                    "Model %s gagal (status %s): %s | Body: %s",
                    model,
                    status_code,
                    e,
                    response_text,
                )
    raise RuntimeError(f"Semua model gagal: {last_error}")


def allowed(update: Update) -> bool:
    return not ALLOWED_USERS or update.effective_user.id in ALLOWED_USERS


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        "Halo! Saya asisten AI. Kirim teks atau gambar untuk menganalisisnya.\n"
        "/reset untuk menghapus memori percakapan."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    history.pop(update.effective_chat.id, None)
    await update.message.reply_text("Memori percakapan dihapus.")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        await update.message.reply_text("Maaf, kamu tidak punya akses ke bot ini.")
        return

    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

    # PERLAKUAN PENGIRIMAN GAMBAR
    if update.message.photo:
        caption = update.message.caption or "Jelaskan isi gambar ini secara detail."

        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()

        base64_image = compress_image(image_bytes)
        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        # Menyusun pesan teks murni dari riwayat sebelumnya
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in list(history[chat_id]):
            messages.append({"role": item["role"], "content": item["content"]})

        # Menambahkan gambar di pesan terbaru
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        })

        # Simpan riwayat sebagai teks biasa
        history[chat_id].append(
            {"role": "user", "content": f"[Pengirim mengirim gambar]: {caption}"}
        )

    # PERLAKUAN PENGIRIMAN TEKS
    else:
        text_input = update.message.text
        history[chat_id].append({"role": "user", "content": text_input})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *list(history[chat_id]),
        ]

    try:
        answer = await ask_ai(messages)
    except Exception as e:
        log.exception("AI Error")
        if history[chat_id]:
            history[chat_id].pop()
        await update.message.reply_text(
            f"Maaf, AI gagal memproses pesan. Detail error: {e}"
        )
        return

    history[chat_id].append({"role": "assistant", "content": answer})

    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(
        MessageHandler((filters.TEXT | filters.PHOTO) & ~filters.COMMAND, chat)
    )

    log.info("Bot berjalan (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
