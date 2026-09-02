import os, re, base64, logging, asyncio, httpx
from aiohttp import web
from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton,
    BotCommand, BotCommandScopeChat, MenuButtonCommands,
)
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
    SCHEDULE_PROMPT_BLOCK, apply_schedule_tag, auto_extract_interests,
    schedule_loop, spawn, weekly_review, weekly_review_loop,
)
from ai_office_shared.shared.media import (
    IMAGE_FILTER, ImageError, addressed_in_group, bot_username, extract_image,
)
from ai_office_shared.shared.photo import (
    PHOTO_HELP, is_photo_request, process_photo, wants_original,
)
from ai_office_shared.shared.imagegen import generate_image
from ai_office_shared.shared.dev_escalation import (
    DEV_FEATURE_PROMPT_BLOCK, parse_dev_feature_tag,
    request_dev_feature, strip_dev_feature_tag,
)
from ai_office_shared.shared.ollama import try_ollama as _try_ollama
from ai_office_shared.shared.auth import (
    deny_message, note_denied_access, office_auth_middleware,
)
from ai_office_shared.shared.web_search import WEB_SEARCH_TOOLS as WEB_SEARCH_TOOL
from ai_office_shared.shared.office import (
    call_office as _call_office_shared, parse_office_tag as _parse_office_tag
)
from ai_office_shared.shared.prompt import enhance_prompt_ex, intent_hint
from ai_office_shared.shared.models import MODEL_SONNET
from ai_office_shared.shared import banter as _banter
from ai_office_shared.shared import group_history as _ghist
from ai_office_shared.shared.identity import roster_prompt, speaker_prompt


async def _call_office(agent_name: str, message: str, user_id: int) -> str:
    return await _call_office_shared(agent_name, message, user_id)


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

== ФОТО ==

Ты УМЕЕШЬ обрабатывать присланные фотографии сама, локально: ретушь лица (убрать
дефекты кожи), фильтры, свет и цвет, кроп, фон, стикеры, апскейл. ЗАПРЕЩЕНО
отвечать «у меня нет инструмента для редактирования изображений» и отправлять
человека в Snapseed или Photoshop — это ложь, обработка встроена. Если просьба
про уже присланное фото пришла текстом — она обрабатывается кодом до тебя;
твоё дело — не отговаривать. Чего действительно нет: рисование новых деталей на
фото (генеративная замена объектов) — про это скажи честно.

Обработки КОМБИНИРУЮТСЯ в одном сообщении: «ретушь и чб» = и ретушь, и
чёрно-белое на одной фотографии, порядок слов не важен. Просьба вдогонку («а
теперь добавь чб») ложится на ПОСЛЕДНИЙ результат, а не на исходник — чтобы
начать заново, говорят «с оригинала» или жмут кнопки под первым фото. Если
спрашивают, умеешь ли ты так, — ответ «да», и покажи пример фразы.

== ОФИСНЫЕ АГЕНТЫ ==

У тебя есть доступ к специалистам AI-офиса. Когда нужны их возможности — добавь в конец ответа тег [OFFICE:ИМЯ:запрос].
ТИЛЛИ — поиск, МИЛЛИ — бизнес, СИЛЛИ — код, ДОКТОР — здоровье, БИЛЛИ — мотивация.
Используй только когда реально нужно. Один тег.""" + SCHEDULE_PROMPT_BLOCK + DEV_FEATURE_PROMPT_BLOCK

LEARN_TRIGGERS = [
    "запомни", "учти", "отныне", "имей в виду", "на будущее",
    "remember that", "note that", "always", "never"
]

async def build_system(user_id: int, sender: str = "") -> str:
    from ai_office_shared.shared.office import instructions_suffix
    notes = await redis_get_notes(redis_client, BOT_NAME_LOWER, user_id)
    result = SYSTEM_BASE
    # Ростер офиса фактом: раньше бот знал коллег только по обрывкам
    # группового чата и путал, кто есть кто.
    result += "\n\n" + roster_prompt(BOT_NAME)
    # А это — кто пишет ИМЕННО СЕЙЧАС. Без этой строки ростер отвечает на
    # вопрос «кто есть кто», но не на вопрос «с кем я говорю», и модель
    # подставляет самого вероятного: 21.08.2026 Крисс начал ответ Яне со
    # слова «Влад».
    who = speaker_prompt(user_id, sender)
    if who:
        result += "\n\n" + who
    if notes:
        result += f"\n\nЗаметки о пользователе:\n{notes}"
    result += await instructions_suffix(redis_client, BOT_NAME_LOWER)
    return result


IMAGE_TRIGGERS = [
    "нарисуй", "нарисуйте", "сгенери", "сгенерируй",
    "покажи картинку", "создай картинку", "draw", "generate image",
    "нарисуй мне", "сделай картинку", "изобрази"
]

def wants_image(text: str) -> bool:
    t = text.lower()
    return any(trigger in t for trigger in IMAGE_TRIGGERS)

# Генерация картинок живёт в shared/imagegen.py: там перевод промпта на
# английский (модели не понимают кириллицу — русская просьба давала картинку
# не по теме), цепочка провайдеров Cloudflare FLUX → Pollinations → Replicate
# и возврат БАЙТОВ вместо ссылки. Прежние _generate_replicate/_generate_
# pollinations удалены — они молча возвращали None и отдавали URL, за которым
# Telegram ходил сам.


async def process_with_image(caption: str, user_id: int, image_b64: str,
                             media_type: str = "image/jpeg", sender: str = "") -> str:
    """Claude vision — анализ изображения. Ollama пропускается (не поддерживает vision)."""
    history = await redis_get_history(redis_client, BOT_NAME_LOWER, user_id)
    system = await build_system(user_id, sender)
    try:
        r = claude.messages.create(
            model=MODEL_SONNET,
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


async def notify_owner(text: str) -> None:
    """Уведомление владельцу. Fail-silent — задача на доске уже создана."""
    try:
        async with httpx.AsyncClient() as c:
            await c.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": YOUR_TELEGRAM_ID, "text": text}, timeout=10
            )
    except Exception as e:
        logger.error(f"notify_owner failed: {e}")

# ── ОБРАБОТКА ФОТО ПО КНОПКАМ ────────────────────────────────────────────────
# Инстаграм режет обработку дневным лимитом. Движок здесь локальный
# (ai_office_shared.shared.photo, Pillow) — лимита нет, ключей не нужно,
# фотографии никуда наружу не уходят.
#
# Картинку в кнопку не положишь: callback_data ограничен 64 байтами. Поэтому
# file_id складываем в Redis под номером сообщения, а в кнопку кладём только
# операцию и номер — по нему потом заново скачиваем оригинал из Telegram.

PHOTO_TTL = 24 * 3600
# «Последнее фото» живёт час: просьба вдогонку приходит через секунды, а через
# сутки «сделай ярче» относилось бы уже к вчерашней картинке.
LAST_PHOTO_TTL = 3600

# Кнопка → фраза для process_photo. Через фразы, а не напрямую в операции:
# разбор живой речи в shared уже покрыт тестами, дублировать его здесь нечем.
PHOTO_ACTIONS = {
    "auto":     ("✨ Улучшить",    "улучши"),
    "portrait": ("🙂 Портрет",     "портрет"),
    # Ретушь и смягчение — РАЗНЫЕ вещи, и до 21.08.2026 это была одна кнопка:
    # «🧴 Ретушь» отправляла фразу, которая вела в пресет «нежное», то есть в
    # размытие всего кадра. Теперь ретушь — отдельная операция (точечно убирает
    # дефекты кожи), а «нежное» осталось фильтром и живёт в меню фильтров.
    "retouch":  ("🧴 Ретушь",      "ретушь"),
    "soft":     ("🕊 Нежное",      "нежное"),
    "vivid":    ("🌈 Сочное",      "сочное"),
    "bw":       ("⚫️ Ч/Б",         "чб"),
    "noir":     ("🎞 Нуар",        "нуар"),
    "vintage":  ("📻 Винтаж",      "винтаж"),
    "film":     ("🎬 Плёнка",      "плёнка"),
    "warm":     ("🔥 Теплее",      "тепло"),
    "cold":     ("❄️ Холоднее",    "холод"),
    "matte":    ("🌫 Матовое",     "матовое"),
    "sharp":    ("🔎 Чётче",       "чётче"),
    "drama":    ("🌩 Драма",       "драма"),
    "nobg":     ("✂️ Убрать фон",  "убери фон"),
    "blurbg":   ("🌁 Размыть фон", "размой фон"),
    "whitebg":  ("⬜️ Белый фон",   "белый фон"),
    "sticker":  ("🏷 Стикер",      "стикер"),
    "square":   ("⬛️ Квадрат",     "квадрат"),
    "story":    ("📱 Стори",       "стори"),
    "avatar":   ("👤 Аватар",      "на аву"),
    "upscale":  ("🔍 Увеличить",   "увеличь"),
    "compress": ("🗜 Сжать",       "сожми"),
}

PHOTO_MENUS = {
    "filters": ["soft", "vivid", "bw", "noir", "vintage", "film", "warm", "cold",
                "matte", "sharp", "drama"],
    "bg":      ["nobg", "blurbg", "whitebg", "sticker"],
    "format":  ["square", "story", "avatar", "upscale", "compress"],
}


def _photo_key(chat_id: int, message_id: int) -> str:
    return f"kriss:photo:{chat_id}:{message_id}"


def _last_photo_key(chat_id: int) -> str:
    return f"kriss:photo:last:{chat_id}"


def _orig_photo_key(chat_id: int) -> str:
    return f"kriss:photo:orig:{chat_id}"


async def remember_photo(chat_id: int, message_id: int, file_id: str,
                         original: bool = False) -> None:
    """
    file_id живёт сутки — дальше кнопки честно скажут прислать фото заново.

    Вторым ключом — «последнее фото в этом чате», на час. Он нужен для самого
    частого сценария: человек кидает фото, а просьбу пишет СЛЕДУЮЩИМ
    сообщением («Кріс, можеш прибрати недоліки на обличчі?»). Без него текст
    приходил в обработчик сообщений, где никакого фото уже нет, и бот отвечал
    про фото так, будто его не видел (21.08.2026).

    Обработанный результат запоминается тем же способом — поэтому «а тепер на
    цій фотографії додай ч/б» ложится НА РЕТУШЬ, а не на исходник. Так было не
    всегда: до 02.09.2026 «последним» оставалось только присланное человеком
    фото, каждая просьба вдогонку начиналась с нуля и предыдущая обработка
    молча пропадала. Исходник при этом не теряется: под ним остались свои
    кнопки, и его помнит отдельный ключ (original=True).
    """
    try:
        await redis_client.setex(_photo_key(chat_id, message_id), PHOTO_TTL, file_id)
        await redis_client.setex(_last_photo_key(chat_id), LAST_PHOTO_TTL,
                                 f"{message_id}:{file_id}")
        if original:
            await redis_client.setex(_orig_photo_key(chat_id), LAST_PHOTO_TTL,
                                     f"{message_id}:{file_id}")
    except Exception as e:
        logger.warning(f"remember_photo failed: {e}")


async def recall_last_photo(chat_id: int) -> tuple:
    """(message_id, file_id) последнего фото в чате или (0, "")."""
    try:
        val = await redis_client.get(_last_photo_key(chat_id))
    except Exception as e:
        logger.warning(f"recall_last_photo failed: {e}")
        return 0, ""
    if not val:
        return 0, ""
    raw = val.decode() if isinstance(val, bytes) else str(val)
    mid, _, file_id = raw.partition(":")
    return (int(mid) if mid.isdigit() else 0), file_id


async def recall_original_photo(chat_id: int) -> tuple:
    """(message_id, file_id) последнего фото, присланного ЧЕЛОВЕКОМ, или (0, "")."""
    try:
        val = await redis_client.get(_orig_photo_key(chat_id))
    except Exception as e:
        logger.warning(f"recall_original_photo failed: {e}")
        return 0, ""
    if not val:
        return 0, ""
    raw = val.decode() if isinstance(val, bytes) else str(val)
    mid, _, file_id = raw.partition(":")
    return (int(mid) if mid.isdigit() else 0), file_id


async def recall_photo(chat_id: int, message_id: int) -> str:
    try:
        val = await redis_client.get(_photo_key(chat_id, message_id))
    except Exception as e:
        logger.warning(f"recall_photo failed: {e}")
        return ""
    if not val:
        return ""
    return val.decode() if isinstance(val, bytes) else str(val)


def _file_id_of(message) -> str:
    """Самый большой размер фото либо документ-картинка."""
    if getattr(message, "photo", None):
        return message.photo[-1].file_id
    doc = getattr(message, "document", None)
    return doc.file_id if doc is not None else ""


def photo_keyboard(mid: int, menu: str = "root") -> InlineKeyboardMarkup:
    """Меню обработки. mid — номер сообщения с оригиналом фото."""
    if menu == "root":
        rows = [
            [InlineKeyboardButton(PHOTO_ACTIONS["auto"][0],     callback_data=f"ph:a:auto:{mid}"),
             InlineKeyboardButton(PHOTO_ACTIONS["portrait"][0], callback_data=f"ph:a:portrait:{mid}")],
            [InlineKeyboardButton(PHOTO_ACTIONS["retouch"][0],  callback_data=f"ph:a:retouch:{mid}"),
             InlineKeyboardButton("🎨 Фильтры", callback_data=f"ph:m:filters:{mid}")],
            [InlineKeyboardButton("✂️ Фон",     callback_data=f"ph:m:bg:{mid}"),
             InlineKeyboardButton("📐 Формат",  callback_data=f"ph:m:format:{mid}")],
            [InlineKeyboardButton("👁 Что на фото", callback_data=f"ph:v:{mid}")],
        ]
        return InlineKeyboardMarkup(rows)

    keys = PHOTO_MENUS.get(menu, [])
    rows, row = [], []
    for key in keys:
        row.append(InlineKeyboardButton(PHOTO_ACTIONS[key][0], callback_data=f"ph:a:{key}:{mid}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"ph:m:root:{mid}")])
    return InlineKeyboardMarkup(rows)


async def send_processed(context, chat_id: int, mid: int, raw: bytes, request: str,
                         reply_markup=None):
    """
    Обработать и отдать результат. Документом, а не фото: Telegram пережимает
    photo и вся обработка становится незаметной.
    """
    res = await process_photo(raw, request)
    if res.error:
        await context.bot.send_message(chat_id=chat_id, text=res.error,
                                       reply_to_message_id=mid)
        return
    stream, filename = res.as_file()
    sent = await context.bot.send_document(
        chat_id=chat_id, document=stream, filename=filename,
        caption=f"✅ {res.caption}", reply_to_message_id=mid,
        reply_markup=reply_markup,
    )

    # Результат — тоже фото, с которым можно работать дальше. Запоминаем его и
    # перевешиваем кнопки на НЕГО: «⚫️ Ч/Б» под отретушированным кадром должна
    # делать чб отретушированного, а не заново тянуть исходник (до 02.09.2026
    # делала именно это — обработка терялась на каждом шаге).
    out_id = getattr(getattr(sent, "document", None), "file_id", "")
    if out_id:
        await remember_photo(chat_id, sent.message_id, out_id)
        if reply_markup is not None:
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id, message_id=sent.message_id,
                    reply_markup=photo_keyboard(sent.message_id))
            except Exception as e:
                # Кнопки остались от исходника — работают, просто на нём.
                logger.warning(f"photo keyboard repoint failed: {e}")

    await log_event(redis_client, BOT_NAME_LOWER, "photo_processed",
                    op=res.op, size=len(res.data or b""))


async def handle_photo_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Кнопки под фото: ph:a:<действие>:<mid> | ph:m:<меню>:<mid> | ph:v:<mid>."""
    q = update.callback_query
    if q.from_user.id not in ALLOWED_USERS:
        await q.answer()
        return

    parts = q.data.split(":")
    kind = parts[1]
    try:
        mid = int(parts[-1])
    except ValueError:
        await q.answer()
        return

    if kind == "m":
        await q.answer()
        await q.edit_message_reply_markup(reply_markup=photo_keyboard(mid, parts[2]))
        return

    chat_id = q.message.chat_id
    file_id = await recall_photo(chat_id, mid)
    if not file_id:
        # Кнопки живут в истории вечно, file_id — сутки. Молчать нельзя.
        await q.answer("Это фото уже не помню — пришли заново 🙂", show_alert=True)
        return

    await q.answer("🎨 Обрабатываю…")
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_document")
        file_obj = await context.bot.get_file(file_id)
        raw = bytes(await file_obj.download_as_bytearray())

        if kind == "v":
            b64 = base64.b64encode(raw).decode()
            answer = await process_with_image("Что на этом фото? Опиши подробно.",
                                              q.from_user.id, b64,
                                              sender=q.from_user.first_name or "")
            await context.bot.send_message(chat_id=chat_id, text=answer,
                                           reply_to_message_id=mid)
            return

        action = PHOTO_ACTIONS.get(parts[2])
        if not action:
            await context.bot.send_message(chat_id=chat_id, text="Не знаю такой кнопки 🤷",
                                           reply_to_message_id=mid)
            return
        # Меню — и под результатом тоже: без него следующий шаг («а теперь чб»)
        # пришлось бы делать кнопкой под ИСХОДНИКОМ, то есть терять ретушь.
        # send_processed перевесит клавиатуру на новое сообщение.
        await send_processed(context, chat_id, mid, raw, action[1],
                             reply_markup=photo_keyboard(mid))
    except Exception as e:
        logger.error(f"handle_photo_callback error: {e}")
        spawn(report_bug("handle_photo_callback crashed", str(e)), name="report_bug")
        await context.bot.send_message(
            chat_id=chat_id, text="⚠️ Не смог обработать. Попробуй ещё раз.",
            reply_to_message_id=mid)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Картинки: фотографией И файлом (Document.IMAGE), в ЛС и в группе.

    До 2026-07-25 хендлер ловил только filters.PHOTO и делал голый `return` на группе —
    картинка файлом пропадала везде, а картинка в группе пропадала молча.
    Теперь единственный «тихий» случай — фото в группе, адресованное не боту.
    """
    if update.effective_user.id not in ALLOWED_USERS:
        return

    is_group = update.effective_chat.type in ["group", "supergroup"]
    if is_group and not addressed_in_group(
        update.message, bot_username(context.bot), BOT_NAME
    ):
        return

    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or str(user_id)

    img = await extract_image(update.message, context.bot)
    if isinstance(img, ImageError):
        logger.warning("handle_photo rejected image: %s", img.reason)
        await update.message.reply_text(img.user_message)
        return

    chat_id = update.effective_chat.id
    mid = update.message.message_id
    file_id = _file_id_of(update.message)
    if file_id:
        await remember_photo(chat_id, mid, file_id, original=True)

    caption = (img.caption or "").strip()
    await log("MSG_IN", f"[фото] {caption or '(без подписи)'}", from_=user_name, to_=BOT_NAME)

    # 1) Просьба обработать («убери фон», «винтаж») — делаем сразу.
    if caption and is_photo_request(caption):
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_document")
        try:
            await send_processed(context, chat_id, mid, base64.b64decode(img.b64), caption,
                                 reply_markup=photo_keyboard(mid))
        except Exception as e:
            logger.error(f"handle_photo processing error: {e}")
            spawn(report_bug("handle_photo processing crashed", str(e)), name="report_bug")
            await update.message.reply_text("⚠️ Не смог обработать фото. Попробуй ещё раз.")
        return

    # 2) Просто фото без подписи — сразу меню, без похода в LLM: Яна кидает
    #    фотки пачками, платить токенами за описание каждой незачем.
    if not caption:
        await update.message.reply_text("Что сделать с фото? 👇",
                                        reply_markup=photo_keyboard(mid))
        return

    # 3) Вопрос про фото — отвечаем как раньше, но кнопки обработки оставляем.
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        response = await process_with_image(caption, user_id, img.b64, img.media_type,
                                            sender=user_name)
        await log("MSG_OUT", f"{BOT_NAME}: {response}", from_=BOT_NAME, to_=user_name)
        await send_long(update, response, reply_markup=photo_keyboard(mid))
    except Exception as e:
        logger.error(f"handle_photo error: {e}")
        spawn(report_bug("handle_photo crashed", str(e)), name="report_bug")
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


def _extract_text(content_blocks) -> str:
    """
    Текст ответа Claude. При веб-поиске берём то, что сказано ПОСЛЕ поиска.

    Раньше склеивались ВСЕ текстовые блоки. При server-side web_search их
    несколько: сначала «сейчас поищу актуальные данные», потом сам поиск,
    потом настоящий ответ. Склейка давала заикание, которое читается как
    сбой бота: «Хороший вопрос... Давай по-честному. Окей, данные есть.
    Погнали — это уже не про алгоритмы...» (чат 15.08). Фраза «сейчас поищу»
    адресована модели, а не читателю.

    Если инструментов не было — поведение прежнее. Если после инструмента
    модель ничего не сказала, отдаём что есть: пустой ответ пользователь
    читает как «бот сломался».
    """
    blocks = list(content_blocks or [])
    last_tool = -1
    for i, b in enumerate(blocks):
        _t = str(getattr(b, "type", "") or "")
        if _t.endswith("tool_use") or _t.endswith("tool_result"):
            last_tool = i

    def _texts(seq):
        return [b.text for b in seq if getattr(b, "text", None)]

    tail = _texts(blocks[last_tool + 1:]) if last_tool >= 0 else _texts(blocks)
    if not tail:
        tail = _texts(blocks)
    return " ".join(tail).strip()


async def process(message: str, user_id: int,
                  sender: str = "", short: bool = False) -> str:
    """sender — кто написал. По HTTP приходил голый текст без автора."""
    # ВАЖНО: в историю идёт ОРИГИНАЛ пользователя, а не вывод enhance-модели.
    # Раньше здесь было `message = await enhance_prompt(...)` — главная модель никогда
    # не видела слов пользователя и на украинском/коротком входе принимала Haiku-артефакт
    # за чужой текст («система сама собі поставила запитання», 2026-07-25).
    enh = await enhance_prompt_ex(
        message, claude_async, redis_client=redis_client, bot_name=BOT_NAME_LOWER,
    )
    await log_event(redis_client, BOT_NAME_LOWER, "message_received",
                    user_id=user_id)
    history = await redis_get_history(redis_client, BOT_NAME_LOWER, user_id)
    history.append({"role": "user",
                    "content": f"[от {sender}] {message}" if sender else message})
    if len(history) > 20:
        history = history[-10:]

    # Уточнение — вторично и живёт в системном промпте, а не в реплике пользователя.
    system = await build_system(user_id, sender) + intent_hint(enh)

    async def _call_claude(msgs):
        """Один чистый вызов Claude с web_search (server-side tool — loop не нужен)."""
        return await claude_async.messages.create(
            model=MODEL_SONNET,
            max_tokens=150 if short else 4096,
            system=system,
            messages=msgs,
            tools=WEB_SEARCH_TOOL
        )

    import re as _re
    try:
        r = await _call_claude(history)
        text = _extract_text(r.content)
        if not text:
            text = "⚠️ Не получил ответ от AI. Попробуй ещё раз."

        # Проверяем обрыв на тексте БЕЗ офисного тега. Системный промпт велит
        # дописывать [OFFICE:ИМЯ:запрос] в КОНЕЦ ответа, а тег кончается на "]",
        # которого нет в списке завершающих символов. Из-за этого законченный
        # ответ считался обрезанным: 15.08 Крисс дописал к «Скажи как удобнее!»
        # ещё один ответ, начинавшийся с «Конечно! Остановился на том, что...» —
        # то есть модель пересказала инструкцию «продолжи», а не продолжила.
        if is_truncated(_re.sub(r'\[OFFICE:[^\]]+\]', '', text).strip()):
            logger.warning(f"Truncated response for {user_id}, retrying...")
            r2 = await _call_claude(history + [
                {"role": "assistant", "content": text},
                {"role": "user",      "content": "Продолжи с того места где остановился."}
            ])
            text = text + " " + _extract_text(r2.content)

        # Офисный агент если Claude запросил
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
        spawn(report_bug(f"BadRequestError (history reset) user={user_id}", str(e)[:300]))
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
        spawn(report_bug("AuthenticationError — ANTHROPIC_API_KEY невалиден!", str(e)[:200]))
        return "⚠️ Ошибка аутентификации AI. Сообщаю администратору."

    except anthropic.PermissionDeniedError as e:
        # 403 — нет доступа к модели или фиче (web_search)
        logger.error(f"PermissionDeniedError: {e}")
        spawn(report_bug("PermissionDeniedError — нет доступа к модели/фиче", str(e)[:200]))
        return "⚠️ Нет доступа к AI. Попробуй позже."

    except anthropic.NotFoundError as e:
        # 404 — неверное имя модели
        logger.error(f"NotFoundError — модель не найдена: {e}")
        spawn(report_bug("NotFoundError — неверное имя модели!", str(e)[:200]))
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
        spawn(report_bug(f"APIError status={status}", str(e)[:300]))
        return "⚠️ Что-то пошло не так с AI. Попробуй ещё раз."

    except Exception as e:
        logger.error(f"process() unexpected error user={user_id}: {type(e).__name__}: {e}")
        spawn(report_bug(f"process() unexpected {type(e).__name__}", str(e)[:300]))
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
                # В общую ленту: Telegram не доставляет сообщения ботов
                # ботам, office:group:history — единственный канал, где
                # коллеги видят, что было сказано.
                await _ghist.push(redis_client, BOT_NAME, text)
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
            await weekly_review(redis_client, BOT_NAME_LOWER, uid, claude_async)
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
        sender  = _banter.sender_of(data) or data.get("sender", "HTTP")
        is_banter = _banter.is_banter(data)
        if not message:
            return web.json_response({"error": "empty message"}, status=400)
        await log("MSG_IN", message, from_=sender, to_=BOT_NAME)
        await log_event(redis_client, BOT_NAME_LOWER, "task_received",
                        user_id=user_id, via="http")
        response = await process(message, user_id,
                                 sender=sender if sender != "HTTP" else "",
                                 short=is_banter)
        if is_banter or data.get("notify", True):
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
    u = update.effective_user
    if u.id not in ALLOWED_USERS:
        # Раньше здесь был голый return: Яна 12.08 нажала /start с нового
        # аккаунта и не получила НИЧЕГО — ни ответа, ни строки в логах.
        await note_denied_access(redis_client, BOT_NAME_LOWER, u.id,
                                 username=u.username or "", first_name=u.first_name or "",
                                 kind="start")
        await update.message.reply_text(deny_message())
        return
    # Приём тоже пишем в лог, не только отказ. Иначе успешный /start остаётся
    # таким же невидимым, каким был отказ до 12.08: человек поздоровался, а в
    # офис-группе пусто — и снова непонятно, дошло вообще или нет.
    who = u.username or u.first_name or str(u.id)
    await log("MSG_IN", "/start", from_=who, to_=BOT_NAME)
    await log_event(redis_client, BOT_NAME_LOWER, "start", user_id=u.id,
                    username=u.username or "")

    from telegram import InlineKeyboardMarkup, InlineKeyboardButton
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("❓ Что умею?", callback_data="kriss_help")
    ]])
    greeting = (
        "Привет! Я Крис — твой персональный ассистент.\n"
        "Помогаю с любыми задачами — и не только.\n\n"
        "Нажми ❓ чтобы узнать что умею, или просто пиши 👇"
    )
    await update.message.reply_text(greeting, reply_markup=kb)
    await log("MSG_OUT", f"{BOT_NAME}: {greeting}", from_=BOT_NAME, to_=who)

# ── МЕНЮ КОМАНД ──────────────────────────────────────────────────────────────
# Синяя кнопка «Меню» в Telegram — это setMyCommands. Без неё базовые функции
# существуют только в голове у того, кто помнит формулировки.

BOT_COMMANDS = [
    ("start",  "Знакомство"),
    ("help",   "Что я умею"),
    ("photo",  "Обработка фото — что умею с картинками"),
    ("menu",   "Быстрые действия"),
    ("tasks",  "Мои задачи и планы"),
    ("status", "Статус: что сделано, что дальше"),
    ("note",   "Запомнить заметку"),
    ("idea",   "Предложить функцию"),
    ("bug",    "Сообщить о проблеме"),
    ("reset",  "Очистить контекст диалога"),
]


async def setup_commands(bot) -> None:
    """
    Меню команд. Fail-soft: сеть до Telegram на старте отваливается регулярно
    (урок #95), а меню — не то, ради чего стоит уронить бота. Не проставилось —
    строка в логе, команды всё равно работают руками.
    """
    cmds = [BotCommand(c, d) for c, d in BOT_COMMANDS]
    try:
        await bot.set_my_commands(cmds)
        await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
        for uid in ALLOWED_USERS:
            try:
                await bot.set_my_commands(cmds, scope=BotCommandScopeChat(chat_id=uid))
            except Exception as e:      # чат с ботом ещё не начат — это норма
                logger.info(f"setup_commands: scope {uid} пропущен ({e})")
        logger.info(f"Меню команд проставлено: {len(cmds)} шт")
    except Exception as e:
        logger.warning(f"setup_commands failed: {e}")


def _allowed(update: Update) -> bool:
    return update.effective_user is not None and update.effective_user.id in ALLOWED_USERS


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    await update.message.reply_text(KRISS_HELP + "\n\n📸 Про фото — /photo")


async def cmd_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    await update.message.reply_text(
        PHOTO_HELP + "\n\nИли просто пришли фото — покажу кнопки.")


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    await update.message.reply_text("Что делаем? 👇", reply_markup=make_task_keyboard())


async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    uid = update.effective_user.id
    notes = await redis_get_notes(redis_client, BOT_NAME_LOWER, uid)
    if not notes:
        await update.message.reply_text(
            "У меня пока нет записей о твоих задачах. Напиши что нужно сделать — запомню.")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await send_long(update, await process("Покажи мой список задач и планов из заметок", uid))


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _allowed(update):
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await send_long(update, await process(
        "Дай краткий статус — что я делал, что запланировано, что в приоритете",
        update.effective_user.id))


async def cmd_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/note текст — сразу в заметки; /note без текста — жду следующее сообщение."""
    if not _allowed(update):
        return
    text = " ".join(context.args or []).strip()
    if not text:
        context.user_data["pending_note"] = True
        await update.message.reply_text("Что запомнить? Пиши следующим сообщением.")
        return
    await redis_add_note(redis_client, BOT_NAME_LOWER, update.effective_user.id, text)
    await update.message.reply_text("✅ Запомнила.")


async def cmd_idea(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Фича-реквест — на доску задач dev-dept, а не сообщением в пустоту."""
    if not _allowed(update):
        return
    text = " ".join(context.args or []).strip()
    if not text:
        context.user_data["pending_feature"] = True
        await update.message.reply_text(
            "Опиши функцию которой не хватает — одним сообщением. Я сразу отправлю Владу.")
        return
    who = update.effective_user.first_name or str(update.effective_user.id)
    task_id = await request_dev_feature(redis_client, BOT_NAME, who, text, notify=notify_owner)
    await update.message.reply_text(
        f"✅ Передала в отдел разработки (задача {task_id})." if task_id
        else "✅ Передала Владу. На доску задач записать не смогла — Redis недоступен.")


async def cmd_bug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Жалоба на поломку — в Bug Lessons группу, где её реально читают."""
    if not _allowed(update):
        return
    text = " ".join(context.args or []).strip()
    if not text:
        context.user_data["pending_bug"] = True
        await update.message.reply_text("Что сломалось? Опиши одним сообщением.")
        return
    who = update.effective_user.first_name or str(update.effective_user.id)
    await report_bug(f"жалоба от {who}", text)
    await notify_owner(f"🐛 {who} через Крис: {text[:500]}")
    await update.message.reply_text("✅ Записала и передала. Спасибо!")


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
    try:
        # httpx-клиент здесь не нужен: поиск идёт server-side инструментом Anthropic.
        r = await claude_async.messages.create(
            model=MODEL_SONNET,
            max_tokens=200,
            system="Ты помощник. Найди прямую ссылку на изображение (заканчивается на .jpg, .jpeg, .png, .gif, .webp) по запросу пользователя. Верни ТОЛЬКО прямой URL картинки, без объяснений. Если не можешь найти — верни слово NONE.",
            messages=[{"role": "user", "content": f"Найди прямую ссылку на картинку: {query}"}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        )
        text = " ".join(b.text for b in r.content if hasattr(b, "text") and b.text).strip()
        if not text or text == "NONE" or len(text) < 10:
            return False
        urls = (re.findall(r'https?://\S+\.(?:jpg|jpeg|png|gif|webp)\S*', text, re.IGNORECASE)
                or re.findall(r'https?://\S+', text))
        if not urls:
            return False
        await update.message.reply_photo(photo=urls[0].rstrip('.,)'))
        return True
    except Exception as e:
        logger.warning(f"find_and_send_photo failed: {e}")
        return False


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ALLOWED_USERS:
        u = update.effective_user
        await note_denied_access(redis_client, BOT_NAME_LOWER, u.id,
                                 username=u.username or "", first_name=u.first_name or "",
                                 kind="message")
        return
    if update.effective_chat.type in ["group", "supergroup"]:
        return

    msg       = update.message.text
    user_id   = update.effective_user.id
    user_name = update.effective_user.first_name or update.effective_user.username or str(user_id)

    await log("MSG_IN", msg, from_=user_name, to_=BOT_NAME)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if is_learn_trigger(msg):
        await redis_add_note(redis_client, BOT_NAME_LOWER, user_id, msg)

    # Просьба вдогонку к только что присланному фото. Самый частый порядок:
    # сначала картинка, потом «прибери недоліки на обличчі» отдельным
    # сообщением. Раньше текст шёл в LLM без картинки, и та отвечала, что
    # редактировать изображения не умеет — при рабочем модуле обработки.
    if is_photo_request(msg):
        chat_id = update.effective_chat.id
        # По умолчанию — поверх последнего результата: «а тепер додай ч/б»
        # означает «к этой фотографии», то есть к той, что на экране.
        # «З оригіналу» — единственный способ сказать «начни заново».
        if wants_original(msg):
            mid, file_id = await recall_original_photo(chat_id)
        else:
            mid, file_id = await recall_last_photo(chat_id)
        if file_id:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id,
                                               action="upload_document")
            try:
                file_obj = await context.bot.get_file(file_id)
                raw = bytes(await file_obj.download_as_bytearray())
                await send_processed(context, update.effective_chat.id, mid, raw, msg,
                                     reply_markup=photo_keyboard(mid))
                await log("MSG_OUT", f"{BOT_NAME}: [обработала последнее фото · {msg}]",
                          from_=BOT_NAME, to_=user_name)
                return
            except Exception as e:
                logger.error(f"handle_message photo follow-up failed: {e}")
                spawn(report_bug("photo follow-up crashed", str(e)), name="report_bug")
                await update.message.reply_text(
                    "⚠️ Фото помню, но обработать не смогла — пришли его ещё раз.")
                return

    if wants_image(msg):
        note = await update.message.reply_text("🎨 Рисую, подожди...")
        await context.bot.send_chat_action(chat_id=update.effective_chat.id,
                                           action="upload_photo")
        res = await generate_image(msg)
        try:
            await note.delete()
        except Exception:
            pass
        if res.ok:
            # Байты, а не ссылка: за ссылкой Telegram ходил сам и падал на
            # медленном провайдере уже вне нашего кода.
            await update.message.reply_photo(photo=res.as_file(), caption=res.caption)
            await log_event(redis_client, BOT_NAME_LOWER, "image_generated",
                            provider=res.provider, size=len(res.data or b""))
            await log("MSG_OUT", f"{BOT_NAME}: [картинка · {res.provider}]",
                      from_=BOT_NAME, to_=user_name)
        else:
            await update.message.reply_text(res.error)
        return

    if wants_photo_search(msg):
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
        success = await find_and_send_photo(update, msg)
        if success:
            await log("MSG_OUT", f"{BOT_NAME}: [фото по запросу]", from_=BOT_NAME, to_=user_name)
            return
        # Если не нашли — идём в обычный process() который объяснит

    # ── Feature request ──────────────────────────────────────────────────────
    # Раньше уходило простым сообщением в личку мимо доски задач и терялось.
    # Теперь — задача на канонический taskboard (assignee=dev-dept) + уведомление.
    if context.user_data.pop("pending_note", False):
        await redis_add_note(redis_client, BOT_NAME_LOWER, user_id, msg)
        await send_long(update, "✅ Запомнила.")
        return

    if context.user_data.pop("pending_bug", False):
        await report_bug(f"жалоба от {user_name}", msg)
        await notify_owner(f"🐛 {user_name} через Крис: {msg[:500]}")
        await send_long(update, "✅ Записала и передала. Спасибо!")
        return

    if context.user_data.get("pending_feature"):
        context.user_data.pop("pending_feature", None)
        task_id = await request_dev_feature(
            redis_client, BOT_NAME, user_name, msg, notify=notify_owner,
        )
        response = (
            f"✅ Передал в отдел разработки (задача {task_id})."
            if task_id else
            "✅ Передал Владу. На доску задач записать не смог — Redis недоступен."
        )
        await send_long(update, response)
        return
    # ─────────────────────────────────────────────────────────────────────────

    response = await process(msg, user_id)
    spawn(auto_extract_interests(redis_client, BOT_NAME_LOWER, user_id, msg, claude_async),
          name="auto_extract_interests")

    # Напоминания из разговора (развязка тега — в shared, одна на все боты)
    response, listing = await apply_schedule_tag(redis_client, BOT_NAME_LOWER, user_id, response)
    if listing:
        await send_long(update, listing)
        return

    # Бот упёрся в отсутствие возможности → задача в dev-dept, а не отписка юзеру.
    feature = parse_dev_feature_tag(response)
    if feature:
        response = strip_dev_feature_tag(response)
        await request_dev_feature(
            redis_client, BOT_NAME, user_name, feature, notify=notify_owner,
        )

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


async def handle_reply(request):
    try:
        data       = await request.json()
        chat_id    = data.get("chat_id")
        text       = data.get("text", "")
        from_agent = data.get("from_agent", "")
        if not chat_id or not text:
            return web.Response(status=400, text="chat_id and text required")
        prefix = f"[{from_agent}] " if from_agent else ""
        from telegram import Bot as _TGBot
        _tg = _TGBot(token=TELEGRAM_TOKEN)
        await _tg.send_message(chat_id=int(chat_id), text=prefix + text)
        return web.Response(text="ok")
    except Exception as e:
        logger.error(f"[КРИСС] /reply error: {e}")
        return web.Response(status=500, text=str(e))





async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Транскрибирует голосовое через Groq Whisper и отвечает как на текст."""
    msg = update.message
    if not msg: return
    uid = update.effective_user.id
    await context.bot.send_chat_action(msg.chat_id, "typing")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        await msg.reply_text("GROQ_API_KEY не настроен 🔑"); return
    try:
        import httpx as _hx
        file_obj = await context.bot.get_file(msg.voice.file_id)
        fp = file_obj.file_path or ""
        url = fp if fp.startswith("http") else f"https://api.telegram.org/file/bot{context.bot.token}/{fp}"
        async with _hx.AsyncClient(timeout=15) as c:
            audio_data = (await c.get(url)).content
        async with _hx.AsyncClient(timeout=30) as c:
            resp = await c.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {groq_key}"},
                files={"file": ("voice.ogg", audio_data, "audio/ogg")},
                data={"model": "whisper-large-v3", "language": "ru"},
            )
            resp.raise_for_status()
            transcription = resp.json().get("text", "").strip()
        if not transcription:
            await msg.reply_text("Не удалось распознать голос 🤷"); return
        reply = await process(transcription, uid)
        if reply:
            await msg.reply_text(reply)
    except Exception as e:
        logger.error(f"[КРИСС voice] {e}")
        await msg.reply_text("Ошибка обработки голоса 🙈")

# handle_group_bot_message удалён: Telegram не доставляет боту сообщения других
# ботов, шанс 20% не разыгрывался ни разу. Межботовые реплики идут по HTTP /task
# (ai_office_shared.shared.banter), их оркестрирует Филли.

async def main():
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=False)
    logger.info("Redis connected")

    app_http = web.Application(middlewares=[office_auth_middleware])
    app_http.router.add_post("/task",           handle_task)
    app_http.router.add_post("/send_scheduled", handle_send_scheduled)
    app_http.router.add_get("/health",          lambda r: web.json_response({"status":"ok","bot":"крисс"}))
    app_http.router.add_post("/reply",          handle_reply)
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
    ptb.add_handler(CommandHandler("help",   cmd_help))
    ptb.add_handler(CommandHandler("photo",  cmd_photo))
    ptb.add_handler(CommandHandler("menu",   cmd_menu))
    ptb.add_handler(CommandHandler("tasks",  cmd_tasks))
    ptb.add_handler(CommandHandler("status", cmd_status))
    ptb.add_handler(CommandHandler("note",   cmd_note))
    ptb.add_handler(CommandHandler("idea",   cmd_idea))
    ptb.add_handler(CommandHandler("bug",    cmd_bug))
    # Кнопки обработки фото — ДО общего CallbackQueryHandler ниже, иначе их
    # съедает он и Яна получает «Хорошо.» вместо обработанной картинки.
    ptb.add_handler(CallbackQueryHandler(handle_photo_callback, pattern="^ph:"))
    # IMAGE_FILTER = PHOTO | Document.IMAGE — картинка файлом раньше не ловилась вовсе.
    ptb.add_handler(MessageHandler(IMAGE_FILTER, handle_photo))
    ptb.add_handler(MessageHandler(filters.VOICE, handle_voice))
    ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    ptb.add_handler(MessageReactionHandler(handle_reaction))
    ptb.add_handler(CallbackQueryHandler(handle_callback))
    async with ptb:
        await ptb.start()
        await ptb.updater.start_polling(drop_pending_updates=True, allowed_updates=["message", "edited_message", "message_reaction", "callback_query"])
        await setup_commands(ptb.bot)
        logger.info("Крис запущен ✅")
        # spawn() удерживает ссылку: голый create_task позволял GC собрать цикл.
        spawn(weekly_review_loop(redis_client, BOT_NAME_LOWER, claude_async),
              name="weekly_review_loop")
        spawn(schedule_loop(redis_client, BOT_NAME_LOWER, ptb.bot), name="schedule_loop")
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())




