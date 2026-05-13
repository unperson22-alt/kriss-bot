import os, logging, asyncio, httpx
from aiohttp import web
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
YOUR_TELEGRAM_ID = int(os.environ["YOUR_TELEGRAM_ID"])
OFFICE_CHAT_ID   = os.environ.get("OFFICE_CHAT_ID", "")
LOG_BOT_URL      = os.environ.get("LOG_BOT_URL", "")
HTTP_PORT        = 8080

# ALLOWED_USERS: comma-separated IDs, falls back to YOUR_TELEGRAM_ID
_raw = os.environ.get("ALLOWED_USERS", "")
ALLOWED_USERS = set(int(x.strip()) for x in _raw.split(",") if x.strip()) if _raw else {YOUR_TELEGRAM_ID}

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
conversation_history = {}

SYSTEM = "Ты — Крис, персональный ИИ-ассистент. Помогаешь с любыми вопросами: отвечаешь, объясняешь, советуешь, помогаешь с задачами. За твоей спиной — AI-офис со специализированными агентами, которых ты можешь привлекать при необходимости. Общаешься неформально, по делу, на русском."

async def log(event: str, msg: str):
    if not LOG_BOT_URL:
        return
    try:
        async with httpx.AsyncClient() as c:
            await c.post(f"{LOG_BOT_URL}/log", json={"agent": "Крис", "type": event, "message": msg}, timeout=5)
    except Exception:
        pass

async def send_to_group(text: str):
    if not OFFICE_CHAT_ID:
        return
    try:
        async with httpx.AsyncClient() as c:
            await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": OFFICE_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        logger.error(f"send_to_group failed: {e}")

async def process(message: str, user_id: int) -> str:
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": "user", "content": message})
    if len(conversation_history[user_id]) > 20:
        conversation_history[user_id] = conversation_history[user_id][-10:]
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
        system=SYSTEM, messages=conversation_history[user_id])
    text = r.content[0].text
    conversation_history[user_id].append({"role": "assistant", "content": text})
    return text

async def handle_task(request):
    data = await request.json()
    message = data.get("message", "")
    user_id = data.get("user_id", YOUR_TELEGRAM_ID)
    await log("MSG_IN", f"[HTTP] {message[:80]}")
    response = await process(message, user_id)
    await send_to_group(f"Крис:\n{response}")
    await log("MSG_OUT", f"Крис: {response[:80]}")
    return web.json_response({"status": "ok", "response": response})

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USERS:
        return
    if update.effective_chat.type in ["group", "supergroup"]:
        return
    msg = update.message.text
    await log("MSG_IN", msg[:80])
    response = await process(msg, update.effective_user.id)
    await log("MSG_OUT", f"Крис: {response[:80]}")
    await update.message.reply_text(response)

async def main():
    app_http = web.Application()
    app_http.router.add_post("/task", handle_task)
    runner = web.AppRunner(app_http)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", HTTP_PORT).start()
    logger.info(f"HTTP on :{HTTP_PORT}")
    ptb = Application.builder().token(TELEGRAM_TOKEN).build()
    ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    async with ptb:
        await ptb.start()
        await ptb.updater.start_polling(drop_pending_updates=True)
        logger.info("Крис запущен")
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())

