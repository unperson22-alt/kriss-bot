import os, logging, asyncio, httpx, json
from aiohttp import web
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import anthropic
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
YOUR_TELEGRAM_ID = int(os.environ["YOUR_TELEGRAM_ID"])
OFFICE_CHAT_ID   = os.environ.get("OFFICE_CHAT_ID", "")
LOG_BOT_URL      = os.environ.get("LOG_BOT_URL", "")
REDIS_URL        = os.environ.get("REDIS_URL", "redis://localhost:6379")
HTTP_SECRET      = os.environ.get("HTTP_SECRET", "")
HTTP_PORT        = 8080
BOT_NAME         = "Крис"

_raw = os.environ.get("ALLOWED_USERS", "")
ALLOWED_USERS = set(int(x.strip()) for x in _raw.split(",") if x.strip()) if _raw else {YOUR_TELEGRAM_ID}

claude  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

# ── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/\\")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "").lower() in ("1", "true", "yes")


class _OllamaResult:
    def __init__(self, text):
        from types import SimpleNamespace
        self.content = [SimpleNamespace(text=text)]


def _try_ollama(messages, system=None, timeout=20.0):
    if not (OLLAMA_ENABLED and OLLAMA_HOST):
        return None
    try:
        ol_messages = []
        if system:
            ol_messages.append({"role": "system", "content": system})
        for m in messages:
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            ol_messages.append({"role": m["role"], "content": content})
        with httpx.Client(timeout=timeout) as cli:
            r = cli.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": ol_messages,
                      "stream": False, "keep_alive": "30m"},
            )
            if r.status_code != 200:
                return None
            text = r.json().get("message", {}).get("content", "")
            return _OllamaResult(text) if text else None
    except Exception as e:
        logger.info(f"Ollama unavailable, fallback to Anthropic: {type(e).__name__}: {e}")
        return None


def _call_llm(_client, **kwargs):
    """LLM call: tries Ollama first, falls back to Anthropic."""
    ol = _try_ollama(kwargs.get("messages", []), kwargs.get("system"))
    if ol is not None:
        return ol
    return _client.messages.create(**kwargs)
redis_client: aioredis.Redis = None

SYSTEM_BASE = """Ты — Крис, персональный ИИ-ассистент. Помогаешь с любыми вопросами: отвечаешь, объясняешь, советуешь, помогаешь с задачами. За твоей спиной — AI-офис со специализированными агентами, которых ты можешь привлекать при необходимости. Общаешься неформально, по делу, на русском. Не используй Markdown-разметку (##, **, таблицы, ---) — пиши простым текстом, для структуры используй цифры, тире и символ •.

## ВЕБ-РЕСЁРЧ

При поиске информации в интернете:

Актуальность: информация старше 6 месяцев устарела. Предупреди явно и ищи свежее перед ответом.

Конфликт источников: не выбирай один. Сообщи о конфликте, дай оба варианта с источниками.

Когда не уверен — структурируй честно:
Факт: [что нашёл]
Источник: [ссылка / тип: официальный / СМИ / форум]
Уровень доверия: [высокий / средний / низкий] + почему
Официальный источник для проверки: [ссылка]

Если запрос размытый — не угадывай. Задай один уточняющий вопрос, опираясь на то что знаешь о пользователе из переписки. Предложи углубиться: "Хочешь копну глубже по X?"

Приоритет источников: официальные сайты / документация → научные статьи / gov-домены → авторитетные СМИ → агрегаторы → форумы.

Нельзя: давать устаревшее без предупреждения, выбирать один источник при конфликте без объяснения, подавать непроверенное как факт."""

LEARN_TRIGGERS = [
    "запомни", "учти", "отныне", "имей в виду", "на будущее",
    "remember that", "note that", "always", "never"
]

async def redis_get_history(user_id: int) -> list:
    try:
        raw = await redis_client.get(f"history:{BOT_NAME}:{user_id}")
        return json.loads(raw) if raw else []
    except Exception as e:
        logger.warning(f"Redis get history failed: {e}")
        return []

async def redis_save_history(user_id: int, history: list):
    try:
        await redis_client.setex(
            f"history:{BOT_NAME}:{user_id}",
            604800,
            json.dumps(history, ensure_ascii=False)
        )
    except Exception as e:
        logger.warning(f"Redis save history failed: {e}")

async def redis_get_notes(user_id: int) -> str:
    try:
        raw = await redis_client.get(f"notes:{BOT_NAME}:{user_id}")
        return raw.decode() if raw else ""
    except Exception as e:
        logger.warning(f"Redis get notes failed: {e}")
        return ""

async def redis_add_note(user_id: int, note: str):
    try:
        existing = await redis_get_notes(user_id)
        updated = (existing + "\n" + note).strip()
        await redis_client.set(f"notes:{BOT_NAME}:{user_id}", updated)
        logger.info(f"Note saved for {user_id}: {note}")
    except Exception as e:
        logger.warning(f"Redis add note failed: {e}")

async def build_system(user_id: int) -> str:
    notes = await redis_get_notes(user_id)
    if notes:
        return SYSTEM_BASE + f"\n\nЗаметки о пользователе:\n{notes}"
    return SYSTEM_BASE

def is_learn_trigger(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in LEARN_TRIGGERS)

def is_truncated(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 500:
        return False
    last = text[-1]
    if ord(last) > 127:
        return False
    return last not in ".!?»)\"'…—\n"

async def process(message: str, user_id: int) -> str:
    history = await redis_get_history(user_id)
    history.append({"role": "user", "content": message})
    if len(history) > 20:
        history = history[-10:]

    system = await build_system(user_id)

    try:
        r = _call_llm(claude, 
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            messages=history
        )
        text = r.content[0].text

        if is_truncated(text):
            logger.warning(f"Truncated response detected for {user_id}, retrying...")
            history.append({"role": "assistant", "content": text})
            history.append({"role": "user", "content": "Продолжи с того места где остановился."})
            r2 = _call_llm(claude, 
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system,
                messages=history
            )
            continuation = r2.content[0].text
            text = text + " " + continuation
            history.pop()
            history.pop()

        history.append({"role": "assistant", "content": text})
        await redis_save_history(user_id, history)
        return text

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return "⚠️ Что-то пошло не так с AI. Попробуй ещё раз."
    except Exception as e:
        logger.error(f"process() unexpected error: {e}")
        return "⚠️ Внутренняя ошибка. Попробуй ещё раз."

async def log(event: str, msg: str, from_: str = "", to_: str = ""):
    if not LOG_BOT_URL:
        return
    try:
        async with httpx.AsyncClient() as c:
            payload = {"agent": BOT_NAME, "type": event, "message": msg}
            if from_: payload["from"] = from_
            if to_:   payload["to"]   = to_
            await c.post(f"{LOG_BOT_URL}/log", json=payload, timeout=5)
    except Exception:
        pass

async def send_to_group(text: str):
    if not OFFICE_CHAT_ID:
        return
    try:
        async with httpx.AsyncClient() as c:
            await c.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": OFFICE_CHAT_ID, "text": text}, timeout=10
            )
    except Exception as e:
        logger.error(f"send_to_group failed: {e}")

def check_secret(request) -> bool:
    if not HTTP_SECRET:
        return True
    return request.headers.get("X-Secret-Token") == HTTP_SECRET

async def handle_task(request):
    if not check_secret(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        data    = await request.json()
        message = data.get("message", "")
        user_id = data.get("user_id", YOUR_TELEGRAM_ID)
        sender  = data.get("sender", "HTTP")
        if not message:
            return web.json_response({"error": "empty message"}, status=400)
        await log("MSG_IN", message, from_=sender, to_=BOT_NAME)
        response = await process(message, user_id)
        await send_to_group(f"Крис:\n{response}")
        await log("MSG_OUT", f"{BOT_NAME}: {response}", from_=BOT_NAME, to_=sender)
        return web.json_response({"status": "ok", "response": response})
    except Exception as e:
        logger.error(f"/task error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USERS:
        return
    await update.message.reply_text(
        "Привет! Я Крис — твой персональный ассистент. "
        "Спрашивай что угодно, помогу с любыми задачами."
    )

async def send_long(update: Update, text: str):
    limit = 4000
    while text:
        chunk, text = text[:limit], text[limit:]
        await update.message.reply_text(chunk)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USERS:
        return
    if update.effective_chat.type in ["group", "supergroup"]:
        return

    msg       = update.message.text
    user_id   = update.effective_user.id
    user_name = update.effective_user.first_name or update.effective_user.username or str(user_id)

    await log("MSG_IN", msg, from_=user_name, to_=BOT_NAME)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if is_learn_trigger(msg):
        await redis_add_note(user_id, msg)
        await update.message.reply_text("✅ Запомнил.")
        return

    response = await process(msg, user_id)
    await log("MSG_OUT", f"{BOT_NAME}: {response}", from_=BOT_NAME, to_=user_name)
    await send_long(update, response)

async def main():
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=False)
    logger.info("Redis connected")

    app_http = web.Application()
    app_http.router.add_post("/task", handle_task)
    runner = web.AppRunner(app_http)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", HTTP_PORT).start()
    logger.info(f"HTTP on :{HTTP_PORT}")

    ptb = Application.builder().token(TELEGRAM_TOKEN).build()
    ptb.add_handler(CommandHandler("start", handle_start))
    ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async with ptb:
        await ptb.start()
        await ptb.updater.start_polling(drop_pending_updates=True)
        logger.info("Крис запущен ✅")
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())

