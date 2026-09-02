"""
Тесты кнопок обработки фото (bot.py).

Проверяются ровно те места, где кнопки ломаются молча и это видно только в
проде: разбор callback_data, потеря file_id через сутки, decode_responses=False
у Redis (get отдаёт bytes, а get_file ждёт строку) и то, что результат уходит
документом, а не пережатым фото.

ЗАПУСК (без сети и без токенов):
    TELEGRAM_TOKEN=x ANTHROPIC_API_KEY=x YOUR_TELEGRAM_ID=1 \
        python -m unittest test_photo_buttons -v
"""
import io
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("TELEGRAM_TOKEN", "x")
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("YOUR_TELEGRAM_ID", "1")

import bot  # noqa: E402

CHAT_ID = 555
MID = 42
FILE_ID = "AgACAgIAAxkBAAaBcTest_FileId_Example_0123456789"


def _jpeg(size=(300, 200)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    im = Image.new("RGB", size)
    px = im.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (x % 256, y % 256, (x + y) % 256)
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


class FakeRedis:
    """Как настоящий у Крисс: decode_responses=False, значения — bytes."""

    def __init__(self):
        self.store = {}

    async def setex(self, key, ttl, value):
        self.store[key] = value.encode() if isinstance(value, str) else value

    async def get(self, key):
        return self.store.get(key)


def _context(raw: bytes):
    ctx = MagicMock()
    ctx.bot = MagicMock()
    ctx.bot.send_chat_action = AsyncMock()
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_document = AsyncMock()
    file_obj = MagicMock()
    file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(raw))
    ctx.bot.get_file = AsyncMock(return_value=file_obj)
    return ctx


def _query(data: str, user_id: int = 1):
    q = MagicMock()
    q.data = data
    q.from_user.id = user_id
    q.message.chat_id = CHAT_ID
    q.answer = AsyncMock()
    q.edit_message_reply_markup = AsyncMock()
    upd = MagicMock()
    upd.callback_query = q
    return upd, q


class PhotoButtonsTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.raw = _jpeg()
        bot.redis_client = FakeRedis()
        bot.ALLOWED_USERS = {1}
        # log_event ходит в Redis по своей схеме — здесь он не предмет теста.
        self._log_event = bot.log_event
        bot.log_event = AsyncMock()
        await bot.remember_photo(CHAT_ID, MID, FILE_ID)

    async def asyncTearDown(self):
        bot.log_event = self._log_event

    async def test_file_id_survives_bytes_redis(self):
        self.assertEqual(await bot.recall_photo(CHAT_ID, MID), FILE_ID)

    async def test_filter_button_sends_processed_document(self):
        ctx = _context(self.raw)
        upd, q = _query(f"ph:a:vintage:{MID}")
        await bot.handle_photo_callback(upd, ctx)

        ctx.bot.send_document.assert_awaited_once()
        kwargs = ctx.bot.send_document.await_args.kwargs
        self.assertEqual(kwargs["chat_id"], CHAT_ID)
        self.assertEqual(kwargs["reply_to_message_id"], MID)
        data = kwargs["document"].getvalue()
        self.assertTrue(data.startswith(b"\xff\xd8"), "результат должен быть JPEG")
        self.assertNotEqual(data, self.raw, "фильтр ничего не изменил")
        # Именно документом: send_photo пережал бы обработку в кашу.
        ctx.bot.send_photo.assert_not_called()

    async def test_every_button_in_every_menu_works(self):
        for menu in ("root", "filters", "bg", "format"):
            for row in bot.photo_keyboard(MID, menu).inline_keyboard:
                for button in row:
                    kind = button.callback_data.split(":")[1]
                    if kind != "a":
                        continue
                    ctx = _context(self.raw)
                    upd, _ = _query(button.callback_data)
                    await bot.handle_photo_callback(upd, ctx)
                    sent = ctx.bot.send_document.await_count
                    said = ctx.bot.send_message.await_count
                    # Либо файл, либо объяснение — молчания быть не должно.
                    self.assertEqual(sent + said, 1, button.callback_data)

    async def test_submenu_button_only_swaps_keyboard(self):
        ctx = _context(self.raw)
        upd, q = _query(f"ph:m:filters:{MID}")
        await bot.handle_photo_callback(upd, ctx)
        q.edit_message_reply_markup.assert_awaited_once()
        ctx.bot.send_document.assert_not_awaited()
        ctx.bot.get_file.assert_not_awaited()

    async def test_expired_photo_explains_instead_of_crashing(self):
        ctx = _context(self.raw)
        upd, q = _query(f"ph:a:auto:{MID + 1}")        # такого фото в Redis нет
        await bot.handle_photo_callback(upd, ctx)
        q.answer.assert_awaited()
        self.assertTrue(q.answer.await_args.kwargs.get("show_alert"))
        ctx.bot.send_document.assert_not_awaited()

    async def test_stranger_gets_nothing(self):
        ctx = _context(self.raw)
        upd, q = _query(f"ph:a:auto:{MID}", user_id=999)
        await bot.handle_photo_callback(upd, ctx)
        ctx.bot.send_document.assert_not_awaited()
        ctx.bot.get_file.assert_not_awaited()

    async def test_broken_callback_data_does_not_raise(self):
        ctx = _context(self.raw)
        upd, _ = _query("ph:a:auto:мусор")
        await bot.handle_photo_callback(upd, ctx)   # не должно бросить
        ctx.bot.send_document.assert_not_awaited()

    def test_all_actions_are_reachable_from_some_menu(self):
        # Кнопка, которой нет ни в одном меню, — мёртвый код.
        reachable = set(sum(bot.PHOTO_MENUS.values(), []))
        for row in bot.photo_keyboard(MID).inline_keyboard:
            for b in row:
                parts = b.callback_data.split(":")
                if parts[1] == "a":
                    reachable.add(parts[2])
        self.assertEqual(reachable, set(bot.PHOTO_ACTIONS))

    def test_retouch_and_soft_are_separate_buttons(self):
        """
        «Ретушь» и «нежное» — разные операции. Одной кнопкой на двоих они были
        до 21.08.2026, и ретушь тогда означала размытие всего кадра.
        """
        from ai_office_shared.shared.photo import parse_request
        self.assertEqual(parse_request(bot.PHOTO_ACTIONS["retouch"][1]).op, "retouch")
        soft = parse_request(bot.PHOTO_ACTIONS["soft"][1])
        self.assertEqual(soft.op, "preset")
        self.assertEqual(soft.preset, "нежное")

    def test_retouch_is_in_the_root_menu(self):
        # Ретушь — то, за чем приходят с селфи: она не должна прятаться в
        # подменю фильтров.
        root = {b.callback_data.split(":")[2]
                for row in bot.photo_keyboard(MID).inline_keyboard for b in row
                if b.callback_data.split(":")[1] == "a"}
        self.assertIn("retouch", root)

    def test_callback_data_fits_telegram_limit(self):
        for menu in ("root", "filters", "bg", "format"):
            for row in bot.photo_keyboard(2 ** 31, menu).inline_keyboard:
                for b in row:
                    self.assertLessEqual(len(b.callback_data.encode()), 64,
                                         b.callback_data)


class PhotoFollowUpTest(unittest.IsolatedAsyncioTestCase):
    """
    Просьба вдогонку: сначала фото, потом текст отдельным сообщением.
    Ровно этот порядок 21.08.2026 привёл к ответу «у мене немає інструменту
    для редагування зображень» при рабочем модуле обработки.
    """

    async def asyncSetUp(self):
        self.raw = _jpeg()
        bot.redis_client = FakeRedis()
        bot.ALLOWED_USERS = {1}
        self._log_event, bot.log_event = bot.log_event, AsyncMock()
        self._log, bot.log = bot.log, AsyncMock()
        await bot.remember_photo(CHAT_ID, MID, FILE_ID)

    async def asyncTearDown(self):
        bot.log_event = self._log_event
        bot.log = self._log

    def _message_update(self, text: str, user_id: int = 1):
        upd = MagicMock()
        upd.effective_user.id = user_id
        upd.effective_user.first_name = "yana"
        upd.effective_user.username = "yana"
        upd.effective_chat.id = CHAT_ID
        upd.effective_chat.type = "private"
        upd.message.text = text
        upd.message.reply_text = AsyncMock()
        return upd

    async def test_last_photo_is_remembered(self):
        self.assertEqual(await bot.recall_last_photo(CHAT_ID), (MID, FILE_ID))

    async def test_no_photo_in_chat_is_not_a_crash(self):
        self.assertEqual(await bot.recall_last_photo(CHAT_ID + 1), (0, ""))

    async def test_ukrainian_follow_up_processes_the_photo(self):
        ctx = _context(self.raw)
        ctx.user_data = {}
        upd = self._message_update("Кріс, можеш будь ласка прибрати недоліки на обличчі?")
        await bot.handle_message(upd, ctx)

        ctx.bot.send_document.assert_awaited_once()
        data = ctx.bot.send_document.await_args.kwargs["document"].getvalue()
        self.assertTrue(data.startswith(b"\xff\xd8"))
        self.assertNotEqual(data, self.raw, "фото вернулось необработанным")

    async def test_ordinary_question_still_goes_to_the_model(self):
        # «Что на фото?» — это вопрос, а не просьба обработать: обработчик
        # текста не должен перехватывать всё подряд.
        ctx = _context(self.raw)
        ctx.user_data = {}
        real_process, bot.process = bot.process, AsyncMock(return_value="это кот")
        try:
            upd = self._message_update("а что вообще на этом фото?")
            await bot.handle_message(upd, ctx)
        finally:
            bot.process = real_process
        ctx.bot.send_document.assert_not_awaited()


class SpeakerPromptTest(unittest.IsolatedAsyncioTestCase):
    """Системный промт обязан называть автора сообщения (инцидент 21.08.2026)."""

    async def asyncSetUp(self):
        bot.redis_client = FakeRedis()

    async def _system_for(self, user_id: int, sender: str = ""):
        async def _no_notes(*a, **kw):
            return ""
        real_notes, bot.redis_get_notes = bot.redis_get_notes, _no_notes
        real_suffix = None
        try:
            import ai_office_shared.shared.office as office
            real_suffix, office.instructions_suffix = office.instructions_suffix, _no_notes
            return await bot.build_system(user_id, sender)
        finally:
            bot.redis_get_notes = real_notes
            if real_suffix is not None:
                office.instructions_suffix = real_suffix

    async def test_yana_is_named_and_vlad_is_excluded(self):
        system = await self._system_for(8993567246, "yana")
        self.assertIn("Яна", system)
        self.assertIn("НЕ Влад", system)

    async def test_owner_gets_his_own_block(self):
        system = await self._system_for(391077101, "Yodka")
        self.assertIn("Влад", system)
        self.assertIn("НЕ Яна", system)

    async def test_photo_capability_is_stated_in_prompt(self):
        # Бот не должен отказываться от того, что умеет.
        self.assertIn("ретушь", bot.SYSTEM_BASE.lower())


if __name__ == "__main__":
    unittest.main()


class FakeTelegram:
    """
    Телеграм, который помнит, что мы ему отдали: send_document возвращает
    сообщение с новым message_id и новым file_id, а get_file по этому file_id
    отдаёт те самые байты. Без этого цепочку обработок проверить нечем —
    моки возвращают MagicMock, и «поверх результата» неотличимо от «поверх
    исходника».
    """

    def __init__(self, files: dict):
        self.files = dict(files)            # file_id -> bytes
        self.next_mid = 100
        self.sent = []                      # (message_id, file_id, caption)
        self.markup_edits = []              # (message_id, markup)
        self.messages = []
        self.downloaded = []                # какие file_id брали в работу

    async def send_chat_action(self, **kw):
        return None

    async def send_message(self, **kw):
        self.messages.append(kw.get("text", ""))
        return MagicMock()

    async def get_file(self, file_id):
        self.downloaded.append(file_id)
        raw = self.files[file_id]
        obj = MagicMock()
        obj.download_as_bytearray = AsyncMock(return_value=bytearray(raw))
        return obj

    async def send_document(self, **kw):
        self.next_mid += 1
        file_id = f"out-{self.next_mid}"
        self.files[file_id] = kw["document"].getvalue()
        self.sent.append((self.next_mid, file_id, kw.get("caption", "")))
        msg = MagicMock()
        msg.message_id = self.next_mid
        msg.document.file_id = file_id
        return msg

    async def edit_message_reply_markup(self, **kw):
        self.markup_edits.append((kw["message_id"], kw["reply_markup"]))
        return MagicMock()


class PhotoChainTest(unittest.IsolatedAsyncioTestCase):
    """
    Просьбы складываются в цепочку: «а тепер на цій фотографії додай ч/б» —
    это чб ПОВЕРХ ретуши, а не чб исходника.

    02.09.2026 Яна получила ровно обратное: ретушь, потом чб оригинала (ретушь
    пропала), потом на прямую просьбу «одночасно і ретуш і чб» — снова одну
    ретушь. Здесь заперты оба места: цепочка в shared и память результата тут.
    """

    async def asyncSetUp(self):
        self.raw = _jpeg()
        bot.redis_client = FakeRedis()
        bot.ALLOWED_USERS = {1}
        self._log_event, bot.log_event = bot.log_event, AsyncMock()
        self._log, bot.log = bot.log, AsyncMock()
        self.tg = FakeTelegram({FILE_ID: self.raw})
        self.ctx = MagicMock()
        self.ctx.bot = self.tg
        self.ctx.user_data = {}
        # Присланное человеком фото — исходник цепочки.
        await bot.remember_photo(CHAT_ID, MID, FILE_ID, original=True)

    async def asyncTearDown(self):
        bot.log_event = self._log_event
        bot.log = self._log

    def _update(self, text: str):
        upd = MagicMock()
        upd.effective_user.id = 1
        upd.effective_user.first_name = "yana"
        upd.effective_chat.id = CHAT_ID
        upd.effective_chat.type = "private"
        upd.message.text = text
        upd.message.reply_text = AsyncMock()
        return upd

    async def test_result_becomes_the_photo_for_the_next_request(self):
        await bot.handle_message(self._update("ретуш"), self.ctx)
        first_mid, first_file, _ = self.tg.sent[-1]
        self.assertEqual(await bot.recall_last_photo(CHAT_ID), (first_mid, first_file))

        await bot.handle_message(self._update("а тепер додай ч/б фільтр"), self.ctx)
        self.assertEqual(len(self.tg.sent), 2)
        # Главное: в работу взяли результат ретуши, а НЕ исходник. Проверка
        # именно по скачанному file_id — «результат отличается от исходника»
        # прошло бы и в сломанном случае: чб оригинала тоже от него отличается.
        self.assertEqual(self.tg.downloaded[-1], first_file)
        self.assertNotIn(FILE_ID, self.tg.downloaded[1:])

    async def test_buttons_under_the_result_work_on_the_result(self):
        await bot.handle_message(self._update("ретуш"), self.ctx)
        new_mid, new_file, _ = self.tg.sent[-1]

        self.assertTrue(self.tg.markup_edits, "клавиатуру не перевесили")
        edited_mid, markup = self.tg.markup_edits[-1]
        self.assertEqual(edited_mid, new_mid)
        for row in markup.inline_keyboard:
            for b in row:
                self.assertTrue(b.callback_data.endswith(f":{new_mid}"), b.callback_data)
        # И этот mid действительно ведёт к результату.
        self.assertEqual(await bot.recall_photo(CHAT_ID, new_mid), new_file)

    async def test_original_is_still_reachable(self):
        await bot.handle_message(self._update("ретуш"), self.ctx)
        await bot.handle_message(self._update("зроби чб з оригіналу"), self.ctx)

        self.assertEqual(len(self.tg.sent), 2)
        self.assertEqual(await bot.recall_original_photo(CHAT_ID), (MID, FILE_ID))
        # «З оригіналу» — единственный способ выйти из цепочки: взяли исходник.
        self.assertEqual(self.tg.downloaded[-1], FILE_ID)

    async def test_result_does_not_overwrite_the_original(self):
        await bot.handle_message(self._update("ретуш"), self.ctx)
        await bot.handle_message(self._update("винтаж"), self.ctx)
        self.assertEqual(await bot.recall_original_photo(CHAT_ID), (MID, FILE_ID))

    async def test_compound_request_is_one_message_with_both_steps(self):
        await bot.handle_message(
            self._update("Мені потрібно на одній фотографії одночасно і ретуш і чб фільтр"),
            self.ctx)
        self.assertEqual(len(self.tg.sent), 1, "два шага — но всё равно один ответ")
        caption = self.tg.sent[-1][2]
        self.assertIn("·", caption, f"подпись не перечислила оба шага: {caption}")
        self.assertLessEqual(len(caption), 1024, "Telegram обрежет такую подпись")

    async def test_button_result_also_carries_its_own_menu(self):
        """
        Кнопка «🧴 Ретушь» → результат с МЕНЮ на себя. Иначе следующий шаг
        человек делает кнопкой под исходником и теряет ретушь — та же потеря,
        что и в текстовом сценарии, только молча.
        """
        q = MagicMock()
        q.data = f"ph:a:retouch:{MID}"
        q.from_user.id = 1
        q.from_user.first_name = "yana"
        q.message.chat_id = CHAT_ID
        q.answer = AsyncMock()
        q.edit_message_reply_markup = AsyncMock()
        upd = MagicMock()
        upd.callback_query = q

        await bot.handle_photo_callback(upd, self.ctx)
        new_mid, new_file, _ = self.tg.sent[-1]
        self.assertEqual(self.tg.markup_edits[-1][0], new_mid)
        self.assertEqual(await bot.recall_photo(CHAT_ID, new_mid), new_file)
