import logging
import os
from collections import defaultdict, deque

import httpx
from dotenv import load_dotenv
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
    raise ValueError("Error: TELEGRAM_BOT_TOKEN belum diset di environment variables!")

if not OPENROUTER_API_KEY:
    raise ValueError("Error: OPENROUTER_API_KEY belum diset di environment variables!")

ALLOWED_USERS = set(
    int(uid.strip())
    for uid in os.getenv("ALLOWED_USERS", "").split(",")
    if uid.strip().isdigit()
)

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Kamu adalah asisten AI Hermes yang ramah, sopan, dan sigap membantu.",
)

history = defaultdict(lambda: deque(maxlen=10))

MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODELS",
        "google/gemini-2.0-flash-lite-001:free,meta-llama/llama-3.3-70b-instruct:free,openrouter/auto",
    ).split(",")
    if m.strip()
]


async def ask_ai(messages: list[dict]) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Telegram AI Bot",
    }
    last_error = None
    async with httpx.AsyncClient(timeout=90) as client:
        for model in MODELS:
            try:
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
                log.warning(
                    "Model %s gagal (status %s): %s", model, status_code, e
                )
    raise RuntimeError(f"Semua model gagal: {last_error}")


def allowed(update: Update) -> bool:
    return not ALLOWED_USERS or update.effective_user.id in ALLOWED_USERS


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        "Halo! Saya asisten AI Hermes. Kirim pesan apa saja.\n/reset untuk hapus memori percakapan."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    history.pop(update.effective_chat.id, None)
    await update.message.reply_text("Memori percakapan dihapus.")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        await update.message.reply_text(
            "Maaf, kamu tidak punya akses ke bot ini."
        )
        return

    chat_id = update.effective_chat.id
    history[chat_id].append({"role": "user", "content": update.message.text})
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[chat_id]]
    try:
        answer = await ask_ai(messages)
    except Exception:
        log.exception("AI error")
        history[chat_id].pop()
        await update.message.reply_text(
            "Maaf, AI sedang sibuk atau kena limit. Coba lagi sebentar."
        )
        return

    history[chat_id].append({"role": "assistant", "content": answer})
    for i in range(0, len(answer), 4000):
        await update.message.reply_text(answer[i : i + 4000])


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
    log.info("Bot berjalan (polling)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
