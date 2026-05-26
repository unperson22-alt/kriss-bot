import re
import os, logging, asyncio, httpx, json, base64
from aiohttp import web
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, MessageHandler, MessageReactionHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
import anthropic
from anthropic import AsyncAnthropic
import redis.asyncio as aioredis
from ai_office_shared.shared.logging import log_event
from ai_office_shared.shared.redis_helpers import (
    redis_get_history, redis_save_history,
    redis_get_notes, redis_add_note,
)
from ai_office_shared.shared.tasks import (
    auto_extract_interests, weekly_review_loop,
    schedule_loop, parse_schedule_tag,
    add_scheduled_task, list_scheduled_tasks,
    remove_scheduled_task, format_task_list,
)
from ai_office_shared.shared.ollama import OllamaResult as _OllamaResult, try_ollama as _try_ollama


# ── Office agents ─────────────────────────────────────────────────────────────

OFFICE_AGENTS = {
    "ТИЛЛИ": {"url": "https://tilly-bot-production.up.railway.app",
               "desc": "веб-поиск, актуальные данные, новости"},
    "МИЛЛИ": {"url": "https://milly-bot-production.up.railway.app",
               "desc": "бизнес, монетизация, стратегия"},
    "СИЛЛИ": {"url": "https://cilly-bot-production.up.railway.app",
               "desc": "код, автоматизация, технические задачи"},
    "ДОКТОР":{"url": "https://dilly-bot-production.up.railway.app",
               "desc": "здоровье, медицинские советы"},
    "БИЛЛИ": {"url": "https://billy-bot-production.up.railway.app",
               "desc": "мотивация, жизненные решения"},
}

async def _enhance_prompt(text: str, client) -> str:
    try:
        r = await client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=200,
            system="Улучши запрос пользователя — чётче и конкретнее. Верни ТОЛЬКО улучшенный запрос.",
            messages=[{"role": "user", "content": text}]
        )
        enhanced = r.content[0].text.strip() if r.content else ""
        return enhanced if enhanced and len(enhanced) > 5 else text
    except Exception:
        return text

async def _call_office(agent_name: str, message: str, user_id: int) -> str:
    info = OFFICE_AGENTS.get(agent_name.upper())
    if not info: return ""
    try:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.post(f"{info['url']}/task",
                json={"message": message, "user_id": user_id})
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception as e:
        logger.warning(f"[office] {agent_name}: {e}")
    return ""

def _parse_office_tag(text: str):
    import re as _re
    m = _re.search(r'\[OFFICE:(\w+):(.+?)\]', text)
    return (m.group(1).upper(), m.group(2).strip()) if m else (None, None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_KEY    = os.environ["ANTHROPIC_API_KEY"]
YOUR_TELEGRAM_ID = int(os.environ["YOUR_TELEGRAM_ID"])
OFFICE_CHAT_ID   = os.environ.get("OFFICE_CHAT_ID", "")
LOG_BOT_URL      = os.environ.get("LOG_BOT_URL", "")
BUG_CHAT_ID      = os.environ.get("BUG_CHAT_ID", "-5197140411")
REDIS_URL        = os.environ.get("REDIS_URL", "redis://localhost:6379")
HTTP_SECRET      = os.environ.get("HTTP_SECRET", "")
HTTP_PORT        = int(os.environ.get("PORT", 8080))
BOT_NAME         = "Крис"
BOT_NAME_LOWER   = "крисс"  # Redis ключ для /metrics (явный override)

# Reactions classification → office:quality:{bot} (feedback loop)
REACTION_UP   = {"👍", "❤️", "🔥", "🥰", "👏", "🎉", "🤩", "🙏"}
REACTION_DOWN = {"👎", "💩", "🤬", "🤮", "😢"}

_raw = os.environ.get("ALLOWED_USERS", "")
ALLOWED_USERS = set(int(x.strip()) for x in _raw.split(",") if x.strip()) if _raw else {YOUR_TELEGRAM_ID}

claude  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
claude_async = AsyncAnthropic(api_key=ANTHROPIC_KEY)

# ── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_HOST    = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/\\")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "").lower() in ("1", "true", "yes")



def _call_llm(_client, **kwargs):
    """LLM call: tries Ollama first, falls back to Anthropic."""
    ol = _try_ollama(kwargs.get("messages", []), kwargs.get("system"))
    if ol is not None:
        return ol
    return _client.messages.create(**kwargs)
redis_client: aioredis.Redis = None


KRISS_HELP = """Вот что я умею 🤖

Персональный ассистент:
— Отвечаю на любые вопросы — объясняю, советую, помогаю
— Веб-поиск в реальном времени — курсы, новости, цены, расписания
— Напоминания и регулярные задачи — просто попроси
— Генерирую изображения по описанию

AI-офис за спиной:
— При необходимости привлекаю специализированных агентов
— Маркетинг, контент, аналитика, трейдинг — всё доступно

И многое другое — просто напиши что нужно 🙂"""
SYSTEM_BASE = """Ты — Крис, персональный ИИ-ассистент. Помогаешь с любыми вопросами: отвечаешь, объясняешь, советуешь, помогаешь с задачами. За твоей спиной — AI-офис со специализированными агентами, которых ты можешь привлекать при необходимости. Общаешься неформально, по делу. Язык — адаптируй под пользователя (пишет по-украински — отвечай по-украински, по-русски — по-русски). ВАЖНО: держи один язык на протяжении ВСЕГО ответа. Не переключайся на другой язык посреди ответа даже если результаты веб-поиска пришли на другом языке — переводи их на язык пользователя. Не используй Markdown-разметку (##, **, таблицы, ---) — пиши простым текстом, для структуры используй цифры, тире и символ •.

## ВЕБ-РЕСЁРЧ

У тебя есть инструмент web_search — РЕАЛЬНЫЙ поиск в интернете. Используй его ОБЯЗАТЕЛЬНО для: курсов валют, новостей, цен, актуальных списков (фильмы/сериалы на платформах), погоды, расписаний, любой информации которая меняется. ЗАПРЕЩЕНО говорить "у меня нет доступа к интернету" — это ложь. ЗАПРЕЩЕНО давать устаревшие списки вместо поиска. Если не уверен — ищи.

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

Нельзя: давать устаревшее без предупреждения, выбирать один источник при конфликте без объяснения, подавать непроверенное как факт.

УПРАВЛЕНИЕ НАПОМИНАНИЯМИ:
Если пользователь просит создать напоминание или регулярное сообщение — добавь в конец своего ответа специальный тег (он невидим для пользователя):
• Каждый день в HH:MM UTC → [SCHEDULE:daily:HH:MM:текст]
• Каждую неделю в день → [SCHEDULE:weekly:mon:HH:MM:текст] (mon/tue/wed/thu/fri/sat/sun)
• Каждые N минут → [SCHEDULE:interval:Nm:текст]
• Один раз в дату → [SCHEDULE:once:YYYY-MM-DD:HH:MM:текст]
• Показать список → [LIST_SCHEDULES]
• Отменить #N → [CANCEL_SCHEDULE:N]
Время всегда в UTC. Если пользователь называет локальное время — переведи в UTC (Германия: UTC+2 летом, UTC+1 зимой).
Отвечай пользователю обычным текстом — подтверди что напоминание создано, укажи когда сработает.

== ОФИСНЫЕ АГЕНТЫ ==

У тебя есть доступ к специалистам AI-офиса. Когда нужны их возможности — добавь в конец ответа тег [OFFICE:ИМЯ:запрос].
ТИЛЛИ — поиск, МИЛЛИ — бизнес, СИЛЛИ — код, ДОКТОР — здоровье, БИЛЛИ — мотивация.
Используй только когда реально нужно. Один тег."""

LEARN_TRIGGERS = [
    "запомни", "учти", "отныне", "имей в виду", "на будущее",
    "remember that", "note that", "always", "never"
]

async def build_system(user_id: int) -> str:
    notes = await redis_get_notes(redis_client, BOT_NAME_LOWER, user_id)
    if notes:
        return SYSTEM_BASE + f"\n\nЗаметки о пользователе:\n{notes}"
    return SYSTEM_BASE


IMAGE_TRIGGERS = [
    "нарисуй", "нарисуйте", "сгенери", "сгенерируй",
    "покажи картинку", "создай картинку", "draw", "generate image",
    "нарисуй мне", "сделай картинку", "изобрази"
]

def wants_image(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in IMAGE_TRIGGERS)

REPLICATE_TOKEN  = os.environ.get("REPLICATE_API_TOKEN", "")

async def _generate_replicate(prompt: str) -> str | None:
    if not REPLICATE_TOKEN:
        return None
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
                headers={"Authorization": f"Bearer {REPLICATE_TOKEN}", "Content-Type": "application/json", "Prefer": "wait"},
                json={"input": {"prompt": prompt, "num_outputs": 1, "output_format": "webp"}}
            )
            if r.status_code in (200, 201):
                data = r.json()
                output = data.get("output")
                if isinstance(output, list) and output:
                    return output[0]
                elif isinstance(output, str):
                    return output
    except Exception as e:
        logger.warning(f"_generate_replicate failed: {e}")
    return None

async def _generate_pollinations(prompt: str) -> str | None:
    try:
        import urllib.parse
        encoded = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(url)
            if r.status_code == 200:
                return url
    except Exception as e:
        logger.warning(f"_generate_pollinations failed: {e}")
    return None

async def generate_image(prompt: str) -> str | None:
    url = await _generate_replicate(prompt)
    if url:
        return url
    return await _generate_pollinations(prompt)

async def process_with_image(caption: str, user_id: int, image_b64: str, media_type: str = "image/jpeg") -> str:
    """Claude vision — анализ изображения. Ollama пропускается (не поддерживает vision)."""
    history = await redis_get_history(redis_client, BOT_NAME_LOWER, user_id)
    system = await build_system(user_id)
    try:
        r = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=history + [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                    {"type": "text", "text": caption or "Что на этом фото? Опиши подробно."}
                ]
            }]
        )
        response = r.content[0].text
        history.append({"role": "user", "content": f"[фото] {caption}"})
        history.append({"role": "assistant", "content": response})
        if len(history) > 20:
            history = history[-10:]
        await redis_save_history(redis_client, BOT_NAME_LOWER, user_id, history)
        return response
    except Exception as e:
        logger.error(f"process_with_image error: {e}")
        return "⚠️ Не смог проанализировать фото. Попробуй ещё раз."

async def report_bug(description: str, error: str = ""):
    """Репорт бага в Bug Lessons группу."""
    if not BUG_CHAT_ID:
        return
    text = f"🐛 [{BOT_NAME}] {description}"
    if error:
        text += f"\nError: {error[:300]}"
    try:
        async with httpx.AsyncClient() as c:
            await c.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": BUG_CHAT_ID, "text": text}, timeout=10
            )
    except Exception as e:
        logger.error(f"report_bug failed: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USERS:
        return
    if update.effective_chat.type in ["group", "supergroup"]:
        return

    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or str(user_id)
    caption = update.message.caption or "Что на этом фото?"

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        file_bytes = await file.download_as_bytearray()
        image_b64 = base64.b64encode(bytes(file_bytes)).decode()
        response = await process_with_image(caption, user_id, image_b64)
        await log("MSG_IN", f"[фото] {caption}", from_=user_name, to_=BOT_NAME)
        await log("MSG_OUT", f"{BOT_NAME}: {response[:80]}", from_=BOT_NAME, to_=user_name)
        await send_long(update, response)
    except Exception as e:
        logger.error(f"handle_photo error: {e}")
        asyncio.create_task(report_bug("handle_photo crashed", str(e)))
        await update.message.reply_text("⚠️ Не смог обработать фото. Попробуй ещё раз.")


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

WEB_SEARCH_TOOL = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]


def _extract_text(content_blocks) -> str:
    """Извлекаем текст из ответа Claude — работает и с tool_use и без."""
    texts = [b.text for b in content_blocks if hasattr(b, "text") and b.text]
    return " ".join(texts).strip()


async def process(message: str, user_id: int) -> str:
    message = await _enhance_prompt(message, claude_async)
    await log_event(redis_client, BOT_NAME_LOWER, "message_received",
                    user_id=user_id)
    history = await redis_get_history(redis_client, BOT_NAME_LOWER, user_id)
    history.append({"role": "user", "content": message})
    if len(history) > 20:
        history = history[-10:]

    system = await build_system(user_id)

    async def _call_claude(msgs):
        """Один чистый вызов Claude с web_search (server-side tool — loop не нужен)."""
        return await claude_async.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            messages=msgs,
            tools=WEB_SEARCH_TOOL
        )

    try:
        r = await _call_claude(history)
        text = _extract_text(r.content)
        if not text:
            text = "⚠️ Не получил ответ от AI. Попробуй ещё раз."

        if is_truncated(text):
            logger.warning(f"Truncated response for {user_id}, retrying...")
            r2 = await _call_claude(history + [
                {"role": "assistant", "content": text},
                {"role": "user",      "content": "Продолжи с того места где остановился."}
            ])
            text = text + " " + _extract_text(r2.content)

        # Офисный агент если Claude запросил
        import re as _re
        agent_name, office_query = _parse_office_tag(text)
        clean_text = _re.sub(r'\[OFFICE:[^\]]+\]', '', text).strip()
        if agent_name and office_query:
            office_result = await _call_office(agent_name, office_query, user_id)
            if office_result:
                synthesis = history + [
                    {"role": "assistant", "content": clean_text},
                    {"role": "user", "content": f"[Данные от {agent_name}]: {office_result}\n\nСинтезируй финальный ответ."}
                ]
                r2 = await _call_claude(synthesis)
                final = _extract_text(r2.content)
                text = final if final else clean_text + f"\n\n📡 {agent_name}: {office_result[:300]}"
            else:
                text = clean_text
        else:
            text = clean_text

        history.append({"role": "assistant", "content": text})
        await redis_save_history(redis_client, BOT_NAME_LOWER, user_id, history)
        await log_event(redis_client, BOT_NAME_LOWER, "response_sent", user_id=user_id)
        return text

    except anthropic.BadRequestError as e:
        # 400 — чаще всего битая/несовместимая история в Redis.
        # Сбрасываем и повторяем с чистого листа.
        logger.error(f"BadRequestError user={user_id}: {e} | Resetting history and retrying.")
        asyncio.create_task(report_bug(f"BadRequestError (history reset) user={user_id}", str(e)[:300]))
        await redis_save_history(redis_client, BOT_NAME_LOWER, user_id, [])
        try:
            r = await _call_claude([{"role": "user", "content": message}])
            text = _extract_text(r.content) or "⚠️ Не получил ответ от AI."
            await redis_save_history(redis_client, BOT_NAME_LOWER, user_id,
                                     [{"role": "user", "content": message},
                                      {"role": "assistant", "content": text}])
            return text
        except Exception as e2:
            logger.error(f"Retry after BadRequestError failed: {e2}")
            return "⚠️ Что-то пошло не так с AI. Попробуй ещё раз."

    except anthropic.AuthenticationError as e:
        # 401 — невалидный ANTHROPIC_API_KEY
        logger.error(f"AuthenticationError — API ключ невалиден! {e}")
        asyncio.create_task(report_bug("AuthenticationError — ANTHROPIC_API_KEY невалиден!", str(e)[:200]))
        return "⚠️ Ошибка аутентификации AI. Сообщаю администратору."

    except anthropic.PermissionDeniedError as e:
        # 403 — нет доступа к модели или фиче (web_search)
        logger.error(f"PermissionDeniedError: {e}")
        asyncio.create_task(report_bug("PermissionDeniedError — нет доступа к модели/фиче", str(e)[:200]))
        return "⚠️ Нет доступа к AI. Попробуй позже."

    except anthropic.NotFoundError as e:
        # 404 — неверное имя модели
        logger.error(f"NotFoundError — модель не найдена: {e}")
        asyncio.create_task(report_bug("NotFoundError — неверное имя модели!", str(e)[:200]))
        return "⚠️ Модель AI не найдена. Сообщаю администратору."

    except anthropic.RateLimitError as e:
        # 429 — превышен лимит запросов
        logger.warning(f"RateLimitError user={user_id}: {e}")
        return "⏳ Слишком много запросов. Попробуй через минуту."

    except anthropic.InternalServerError as e:
        # 5xx — проблема на стороне Anthropic
        logger.error(f"InternalServerError (Anthropic side): {e}")
        return "⚠️ Сервер AI временно недоступен. Попробуй через минуту."

    except anthropic.APIError as e:
        # Всё остальное — неизвестная ошибка API
        status = getattr(e, "status_code", "?")
        logger.error(f"APIError status={status} user={user_id}: {e}")
        await log_event(redis_client, BOT_NAME_LOWER, "api_error", level="error",
                        user_id=user_id, error=str(e)[:200])
        asyncio.create_task(report_bug(f"APIError status={status}", str(e)[:300]))
        return "⚠️ Что-то пошло не так с AI. Попробуй ещё раз."

    except Exception as e:
        logger.error(f"process() unexpected error user={user_id}: {type(e).__name__}: {e}")
        asyncio.create_task(report_bug(f"process() unexpected {type(e).__name__}", str(e)[:300]))
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
        return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": OFFICE_CHAT_ID, "text": text}, timeout=10
            )
            data = r.json()
            if data.get("ok"):
                msg_id = data["result"]["message_id"]
                await remember_my_message(int(OFFICE_CHAT_ID), msg_id)
                return msg_id
    except Exception as e:
        logger.error(f"send_to_group failed: {e}")
    return None


# ── FEEDBACK LOOP: msg owner mapping for reactions ───────────────────────────
async def remember_my_message(chat_id: int, message_id: int):
    """Маркер 'это сообщение наше' для handle_reaction."""
    if not redis_client:
        return
    try:
        await redis_client.setex(
            f"office:msg:{chat_id}:{message_id}",
            86400 * 14,
            BOT_NAME_LOWER.encode()
        )
    except Exception as e:
        logger.warning(f"remember_my_message failed: {e}")

def check_secret(request) -> bool:
    if not HTTP_SECRET:
        return True
    return request.headers.get("X-Secret-Token") == HTTP_SECRET


async def handle_weekly_review(request):
    """Cloudflare Cron вызывает этот endpoint раз в неделю."""
    if not check_secret(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        # Получаем всех пользователей у которых есть история
        keys = []
        async for key in redis_client.scan_iter(f"history:{BOT_NAME}:*"):
            keys.append(key)
        for key in keys:
            uid = int(key.decode().split(":")[-1])
            await weekly_review(uid)
        return web.json_response({"status": "ok", "users": len(keys)})
    except Exception as e:
        logger.error(f"/cron/weekly_review error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_reset_history(request: web.Request) -> web.Response:
    """DELETE /reset_history?user_id=XXX — сброс истории пользователя"""
    user_id_str = request.rel_url.query.get("user_id", "")
    if not user_id_str:
        # Сброс всех пользователей kriss
        if redis_client:
            keys = []
            async for key in redis_client.scan_iter(f"history:{BOT_NAME}:*"):
                keys.append(key)
            if keys:
                await redis_client.delete(*keys)
            return web.json_response({"status": "ok", "deleted": len(keys)})
        return web.json_response({"status": "no_redis"})
    try:
        user_id = int(user_id_str)
        key = f"history:{BOT_NAME}:{user_id}"
        if redis_client:
            await redis_client.delete(key)
        return web.json_response({"status": "ok", "deleted": key})
    except Exception as e:
        return web.json_response({"status": "error", "msg": str(e)})

# ── QUICK REPLY HELPERS ───────────────────────────────────────────────────────
TASK_KEYWORDS = {
    "задач", "сделать", "список", "план", "напомни", "не забудь",
    "дедлайн", "срок", "статус", "готово", "выполнено", "успел",
    "заметка", "запомни", "помни", "учти",
}

def should_show_quick_reply(response: str) -> bool:
    """Показываем кнопки если ответ касается задач/планов/статусов."""
    r = response.lower()
    return any(kw in r for kw in TASK_KEYWORDS)

def make_task_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📋 Задачи",  callback_data="qr_tasks"),
            InlineKeyboardButton("🔄 Статус",  callback_data="qr_status"),
        ],
        [
            InlineKeyboardButton("💬 Заметка", callback_data="qr_note"),
            InlineKeyboardButton("✅ Готово",   callback_data="qr_done"),
            InlineKeyboardButton("💡 Запросить функцию", callback_data="qr_feature"),
        ],
    ])

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик Quick Reply кнопок."""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if query.data == "qr_tasks":
        notes = await redis_get_notes(redis_client, BOT_NAME_LOWER, user_id)
        if notes:
            reply = await process("Покажи мой список задач и планов из заметок", user_id)
        else:
            reply = "У меня пока нет записей о твоих задачах. Напиши что нужно сделать — запомню."

    elif query.data == "qr_status":
        reply = await process("Дай краткий статус — что я делал, что запланировано, что в приоритете", user_id)

    elif query.data == "qr_note":
        reply = "Напиши что запомнить — я сохраню в заметки."

    elif query.data == "qr_done":
        reply = await process("Отмечаю последнее обсуждаемое как выполненное. Что следующее?", user_id)

    elif query.data == "qr_feature":
        context.user_data["pending_feature"] = True
        reply = "Опиши функцию которой не хватает — одним сообщением. Я сразу отправлю Владу."

    else:
        reply = "Хорошо."

    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    sent = await context.bot.send_message(chat_id=query.message.chat_id, text=reply)
    if sent:
        await remember_my_message(sent.chat_id, sent.message_id)

# ── SCHEDULED MESSAGES ────────────────────────────────────────────────────────
async def handle_send_scheduled(request):
    """
    POST /send_scheduled — внешний триггер (Railway Cron) шлёт сообщение от Крисс.
    Body: {"chat_id": int, "message": str, "user_id": int (optional)}
    """
    if not check_secret(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    try:
        data    = await request.json()
        chat_id = data.get("chat_id")
        message = data.get("message", "").strip()
        user_id = data.get("user_id", YOUR_TELEGRAM_ID)
        if not chat_id or not message:
            return web.json_response({"error": "chat_id and message required"}, status=400)
        bot = request.app["bot"]
        # Генерируем через Claude если message — это инструкция а не готовый текст
        if data.get("generate", False):
            response = await process(message, user_id)
            text = response
        else:
            text = message
        sent = await bot.send_message(chat_id=int(chat_id), text=text)
        if sent:
            await remember_my_message(sent.chat_id, sent.message_id)
        await log_event(redis_client, BOT_NAME_LOWER, "scheduled_sent",
                        chat_id=chat_id, length=len(text))
        return web.json_response({"status": "ok", "length": len(text)})
    except Exception as e:
        logger.error(f"/send_scheduled error: {e}")
        return web.json_response({"error": str(e)}, status=500)

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
        await log_event(redis_client, BOT_NAME_LOWER, "task_received",
                        user_id=user_id, via="http")
        response = await process(message, user_id)
        if data.get("notify", True):
            await send_to_group(f"Крис:\n{response}")
        await log("MSG_OUT", f"{BOT_NAME}: {response}", from_=BOT_NAME, to_=sender)
        return web.json_response({"status": "ok", "response": response})
    except Exception as e:
        logger.error(f"/task error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_kriss_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.from_user.id not in ALLOWED_USERS:
        await query.answer(); return
    await query.answer()
    await query.message.reply_text(KRISS_HELP)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USERS:
        return
    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("❓ Что умею?", callback_data="kriss_help")
    ]])
    await update.message.reply_text(
        "Привет! Я Крис — твой персональный ассистент.\n"
        "Помогаю с любыми задачами — и не только.\n\n"
        "Нажми ❓ чтобы узнать что умею, или просто пиши 👇",
        reply_markup=kb
    )

async def send_long(update: Update, text: str, reply_markup=None):
    limit = 4000
    sent = None
    while text:
        chunk, rest = text[:limit], text[limit:]
        kb = reply_markup if not rest else None
        sent = await update.message.reply_text(chunk, reply_markup=kb)
        text = rest
    if sent:
        await remember_my_message(sent.chat_id, sent.message_id)


PHOTO_SEARCH_TRIGGERS = [
    "знайди фото", "знайди картинку", "покажи фото", "покажи картинку",
    "відправ фото", "відправ картинку", "пришли фото", "пришли картинку",
    "хочу фото", "хочу картинку", "фото котик", "фото кот",
    "найди фото", "найди картинку", "покажи картинку", "пришли картинку",
    "send photo", "find photo", "show photo", "find image", "show image",
    "знайди зображення", "покажи зображення"
]

def wants_photo_search(text: str) -> bool:
    t = text.lower()
    # Явные триггеры
    if any(trigger in t for trigger in PHOTO_SEARCH_TRIGGERS):
        return True
    # "фото/картинку/зображення + [чего-то] з інтернету/онлайн"
    if any(w in t for w in ["з інтернету", "из интернета", "онлайн", "online"]):
        if any(w in t for w in ["фото", "картинк", "зображен", "photo", "image", "picture"]):
            return True
    return False

async def find_and_send_photo(update, query: str) -> bool:
    """Ищет фото через web_search и отправляет в чат. Возвращает True если успешно."""
    import re
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            # Bing Image Search через web_search API не доступен напрямую
            # Используем Anthropic web_search чтобы найти прямой URL картинки
            r = await claude_async.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system="Ты помощник. Найди прямую ссылку на изображение (заканчивается на .jpg, .jpeg, .png, .gif, .webp) по запросу пользователя. Верни ТОЛЬКО прямой URL картинки, без объяснений. Если не можешь найти — верни слово NONE.",
                messages=[{"role": "user", "content": f"Найди прямую ссылку на картинку: {query}"}],
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
            )
            texts = [b.text for b in r.content if hasattr(b, "text") and b.text]
            url = " ".join(texts).strip()
            
            if not url or url == "NONE" or len(url) < 10:
                return False
            
            # Извлекаем URL если есть лишний текст
            urls = re.findall(r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)\S*', url, re.IGNORECASE)
            if not urls:
                # Попробуем любой https URL
                urls = re.findall(r'https?://\S+', url)
            
            if not urls:
                return False
            
            photo_url = urls[0].rstrip('.,)')
            await update.message.reply_photo(photo=photo_url)
            return True
    except Exception as e:
        logger.warning(f"find_and_send_photo failed: {e}")
        return False


async def analyze_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, bot_name: str, system_prompt: str) -> None:
    """Анализирует фото через Claude Vision и спрашивает чем помочь."""
    msg = update.message
    caption = msg.caption or ""
    await context.bot.send_chat_action(msg.chat_id, "typing")
    try:
        photo = msg.photo[-1]
        file_obj = await context.bot.get_file(photo.file_id)
        tg_token = context.bot.token
        fp = file_obj.file_path or ""
        photo_url = fp if fp.startswith("http") else ("https://api.telegram.org/file/bot" + tg_token + "/" + fp)
        import httpx as _httpx, base64 as _b64
        async with _httpx.AsyncClient(timeout=15) as c:
            img_r = await c.get(photo_url)
            img_b64 = _b64.b64encode(img_r.content).decode()
        prompt = (
            "Пользователь прислал тебе фото" +
            (". Подпись: " + caption if caption else " без подписи") +
            ". Кратко опиши что видишь (1-2 предложения) и спроси чем можешь помочь."
        )
        import anthropic as _ant
        _client = _ant.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = await asyncio.to_thread(
            _client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        await msg.reply_text(resp.content[0].text)
    except Exception as e:
        await msg.reply_text("Хм, не смог прочитать это фото. Опиши текстом.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USERS:
        return
    if update.effective_chat.type in ["group", "supergroup"]:
        return

    if update.message and update.message.photo:
        await analyze_photo(update, context, BOT_NAME, SYSTEM_PROMPT)
        return
    msg       = update.message.text
    user_id   = update.effective_user.id
    user_name = update.effective_user.first_name or update.effective_user.username or str(user_id)

    await log("MSG_IN", msg, from_=user_name, to_=BOT_NAME)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if is_learn_trigger(msg):
        await redis_add_note(redis_client, BOT_NAME_LOWER, user_id, msg)

    if wants_image(msg):
        await update.message.reply_text("🎨 Рисую, подожди...")
        url = await generate_image(msg)
        if url:
            await update.message.reply_photo(photo=url, caption=f"🎨 {msg[:200]}")
        else:
            await update.message.reply_text("❌ Не получилось нарисовать. Попробуй ещё раз.")
        return

    if wants_photo_search(msg):
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
        success = await find_and_send_photo(update, msg)
        if success:
            await log("MSG_OUT", f"{BOT_NAME}: [фото по запросу]", from_=BOT_NAME, to_=user_name)
            return
        # Если не нашли — идём в обычный process() который объяснит

    # ── Feature request ──────────────────────────────────────────────────────
    if context.user_data.get("pending_feature"):
        context.user_data.pop("pending_feature", None)
        sender_name = update.effective_user.first_name or update.effective_user.username or str(user_id)
        feature_text = "💡 Запрос функции для Крисс\n\nОт: " + sender_name + "\nОписание: " + msg
        try:
            await context.bot.send_message(chat_id=391077101, text=feature_text)
            response = "✅ Отправил Владу! Он рассмотрит."
        except Exception as fe:
            response = "⚠️ Не смог отправить: " + str(fe)
        await send_long(update, response)
        return
    # ─────────────────────────────────────────────────────────────────────────

    response = await process(msg, user_id)
    asyncio.create_task(auto_extract_interests(redis_client, BOT_NAME_LOWER, user_id, msg, claude_async))

    # Обработка тегов шедулера
    tag = parse_schedule_tag(response)
    if tag:
        if tag["action"] == "add":
            await add_scheduled_task(redis_client, BOT_NAME_LOWER, user_id, tag)
        elif tag["action"] == "cancel":
            await remove_scheduled_task(redis_client, BOT_NAME_LOWER, user_id, tag["index"])
        elif tag["action"] == "list":
            tasks = await list_scheduled_tasks(redis_client, BOT_NAME_LOWER, user_id)
            task_text = await format_task_list(tasks)
            await send_long(update, task_text)
            return
        # Убираем тег из текста перед отправкой
        response = re.sub(r'\[(?:SCHEDULE|CANCEL_SCHEDULE|LIST_SCHEDULES)[^\]]*\]', '', response).strip()

    await log("MSG_OUT", f"{BOT_NAME}: {response}", from_=BOT_NAME, to_=user_name)
    keyboard = make_task_keyboard() if should_show_quick_reply(response) else None
    await send_long(update, response, reply_markup=keyboard)


# weekly_review_loop → ai_office_shared.shared.tasks



async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /reset — сброс истории текущего пользователя (только owner может для любого)"""
    caller_id = update.effective_user.id
    
    # Если owner передал user_id аргументом — сбрасываем того пользователя
    args = context.args
    if caller_id in ALLOWED_USERS and args:
        try:
            target_id = int(args[0])
        except:
            target_id = caller_id
    else:
        target_id = caller_id
    
    key = f"history:{BOT_NAME}:{target_id}"
    if redis_client:
        await redis_client.delete(key)
        await update.message.reply_text(f"✅ История сброшена для {target_id}")
    else:
        await update.message.reply_text("⚠️ Redis недоступен")

async def cmd_reset_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /resetall — сброс истории ВСЕХ пользователей"""
    if redis_client:
        keys = []
        async for key in redis_client.scan_iter(f"history:{BOT_NAME}:*"):
            keys.append(key)
        if keys:
            await redis_client.delete(*keys)
        await update.message.reply_text(f"✅ Сброшено {len(keys)} историй")
    else:
        await update.message.reply_text("⚠️ Redis недоступен")

async def handle_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Реакции 👍/👎 → office:quality:{bot} (HASH up/down)."""
    reaction = update.message_reaction
    if not reaction:
        return

    chat_id = reaction.chat.id
    msg_id  = reaction.message_id

    try:
        owner_raw = await redis_client.get(f"office:msg:{chat_id}:{msg_id}")
    except Exception as e:
        logger.warning(f"reaction owner lookup failed: {e}")
        return
    if not owner_raw or owner_raw.decode() != BOT_NAME_LOWER:
        return

    old_emojis = {r.emoji for r in (reaction.old_reaction or []) if getattr(r, "emoji", None)}
    new_emojis = {r.emoji for r in (reaction.new_reaction or []) if getattr(r, "emoji", None)}
    added   = new_emojis - old_emojis
    removed = old_emojis - new_emojis

    delta_up   = sum(1 for e in added if e in REACTION_UP)   - sum(1 for e in removed if e in REACTION_UP)
    delta_down = sum(1 for e in added if e in REACTION_DOWN) - sum(1 for e in removed if e in REACTION_DOWN)

    if delta_up == 0 and delta_down == 0:
        return

    try:
        key = f"office:quality:{BOT_NAME_LOWER}"
        if delta_up:
            await redis_client.hincrby(key, "up", delta_up)
        if delta_down:
            await redis_client.hincrby(key, "down", delta_down)
        logger.info(f"REACTION msg={msg_id} added={added} removed={removed} du={delta_up} dd={delta_down}")
    except Exception as e:
        logger.warning(f"quality hincrby failed: {e}")


async def main():
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=False)
    logger.info("Redis connected")

    app_http = web.Application()
    app_http.router.add_post("/task",           handle_task)
    app_http.router.add_post("/send_scheduled", handle_send_scheduled)
    app_http.router.add_get("/health",          lambda r: web.json_response({"status":"ok","bot":"крисс"}))
    runner = web.AppRunner(app_http)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", HTTP_PORT).start()
    logger.info(f"HTTP on :{HTTP_PORT}")

    ptb = Application.builder().token(TELEGRAM_TOKEN).build()
    app_http["bot"] = ptb.bot
    ptb.add_handler(CommandHandler("start", handle_start))
    ptb.add_handler(CallbackQueryHandler(handle_kriss_help, pattern="^kriss_help$"))
    ptb.add_handler(CommandHandler("reset", cmd_reset))
    ptb.add_handler(CommandHandler("resetall", cmd_reset_all))
    ptb.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    ptb.add_handler(MessageReactionHandler(handle_reaction))
    ptb.add_handler(CallbackQueryHandler(handle_callback))
    async with ptb:
        await ptb.start()
        await ptb.updater.start_polling(drop_pending_updates=True, allowed_updates=["message", "edited_message", "message_reaction", "callback_query"])
        logger.info("Крис запущен ✅")
        asyncio.create_task(weekly_review_loop(redis_client, BOT_NAME_LOWER, claude_async))
        asyncio.create_task(schedule_loop(redis_client, BOT_NAME_LOWER, ptb.bot))
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())




