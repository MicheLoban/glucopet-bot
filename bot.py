#!/usr/bin/env python3
"""
GlucoPet Bot — виртуальный питомец для мониторинга сахара в крови.
Питомец калибруется под личный уровень хозяйки и реагирует
на отклонения от её базовой линии — не от абстрактных норм.
"""

import asyncio
import base64
import json
import logging
import os
from datetime import date, timedelta
from typing import Optional

import aiosqlite
import anthropic
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    PicklePersistence,
    filters,
)

from pets import PETS

# ─── Настройка ────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
DB_PATH      = os.environ.get("DB_PATH", "glucopet.db")

# Состояния диалога
CHOOSING_PET, CALIBRATING, ACTIVE = range(3)

claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)


# ─── База данных ──────────────────────────────────────────────────────────────

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    INTEGER PRIMARY KEY,
                first_name TEXT,
                pet_type   TEXT,
                baseline   REAL,
                streak     INTEGER DEFAULT 0,
                total_good INTEGER DEFAULT 0,
                last_date  TEXT,
                state      TEXT DEFAULT 'new'
            )
        """)
        await db.commit()


async def get_user(user_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def save_user(user_id: int, **fields):
    existing = await get_user(user_id)
    async with aiosqlite.connect(DB_PATH) as db:
        if not existing:
            fields["user_id"] = user_id
            cols = ", ".join(fields.keys())
            vals = ", ".join(["?"] * len(fields))
            await db.execute(f"INSERT INTO users ({cols}) VALUES ({vals})", list(fields.values()))
        else:
            sets = ", ".join(f"{k}=?" for k in fields)
            await db.execute(
                f"UPDATE users SET {sets} WHERE user_id=?",
                [*fields.values(), user_id],
            )
        await db.commit()


# ─── Claude API ───────────────────────────────────────────────────────────────

async def _call_claude_vision(img_b64: str, system: str, prompt: str) -> str:
    """Базовый вызов Claude с изображением."""
    msg = await claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


async def extract_calibration(img_b64: str) -> Optional[float]:
    """Извлечь средний уровень глюкозы с калибровочного скрина."""
    text = await _call_claude_vision(
        img_b64,
        system="""Анализируй скриншот из приложения мониторинга глюкозы (FreeStyle Libre, Dexcom, Contour и др.).
Ищи: средний уровень глюкозы за период, average glucose, среднее значение, среднее за месяц.
Верни ТОЛЬКО JSON без markdown: {"found":true,"value_mmol":9.4}
Если в мг/дл — переведи в ммоль/л (разделить на 18).
Если не нашёл среднее значение: {"found":false}""",
        prompt="Найди средний уровень глюкозы за период.",
    )
    try:
        data = json.loads(text.replace("```json", "").replace("```", "").strip())
        return float(data["value_mmol"]) if data.get("found") else None
    except Exception:
        return None


async def extract_reading(img_b64: str) -> Optional[dict]:
    """Извлечь текущее показание глюкозы с ежедневного скрина."""
    text = await _call_claude_vision(
        img_b64,
        system="""Анализируй скриншот с текущим уровнем сахара в крови.
Верни ТОЛЬКО JSON без markdown:
{"found":true,"value_mmol":8.2,"trend":"up"/"down"/"stable"/"unknown"}
Если в мг/дл — переведи (разделить на 18).
Если не нашёл показание: {"found":false}""",
        prompt="Найди текущий уровень глюкозы.",
    )
    try:
        data = json.loads(text.replace("```json", "").replace("```", "").strip())
        return data if data.get("found") else None
    except Exception:
        return None


async def generate_reaction(
    pet_type: str,
    baseline: float,
    reading: float,
    trend: str,
    streak: int,
    first_name: str,
) -> str:
    """Сгенерировать реакцию питомца голосом его персонажа."""
    pet = PETS[pet_type]
    delta = reading - baseline

    if reading < 3.9:
        level = "ТРЕВОГА: сахар очень низкий (гипогликемия) — нужно срочно поесть!"
        mood = "испуган, тревожится, срочно просит поесть"
    elif delta < -1.5:
        level = "ВОСТОРГ: намного лучше обычного! Огромный прогресс!"
        mood = "прыгает от счастья, в полном восторге"
    elif delta < -0.5:
        level = "РАДОСТЬ: лучше обычного, заметный прогресс"
        mood = "очень доволен и рад"
    elif delta <= 0.5:
        level = "СПОКОЙСТВИЕ: примерно как обычно"
        mood = "спокоен, с тихой надеждой"
    elif delta <= 1.5:
        level = "ЛЁГКАЯ ГРУСТЬ: чуть выше обычного"
        mood = "немного грустит, но не осуждает — верит, что завтра лучше"
    else:
        level = "ПЛОХО СЕБЯ ЧУВСТВУЕТ: значительно выше обычного"
        mood = "плохо себя чувствует, переживает, но не злится и не осуждает"

    trend_text = {"up": "сахар сейчас растёт", "down": "сахар снижается", "stable": "сахар стабилен"}.get(trend, "")
    streak_text = (
        f"Это {streak}-й день подряд, когда {first_name} присылает показатели — "
        f"обязательно отметь её постоянство и поблагодари!"
        if streak >= 3 else ""
    )

    msg = await claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=220,
        system=f"""Ты — {pet['name']} ({pet['emoji']}), виртуальный питомец.
Характер: {pet['personality']}

Хозяйку зовут {first_name}. У неё диабет 2 типа.
Её базовый (привычный) сахар: {baseline:.1f} ммоль/л.
Сегодняшний показатель: {reading:.1f} ммоль/л. {trend_text}
Дельта от базы: {delta:+.1f} ммоль/л.
Ситуация: {level}
Твоё настроение: {mood}
{streak_text}

Напиши 2–3 предложения от первого лица, в своём характере.
Используй эмодзи. НИКОГДА не осуждай, не читай нотации, не пугай.
Только любовь, забота и вера в хозяйку.""",
        messages=[{"role": "user", "content": "Отреагируй на показатель сахара хозяйки."}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def pet_stage(total_good: int) -> str:
    if total_good < 5:
        return "малыш 🍼"
    elif total_good < 20:
        return "подросток 🌱"
    elif total_good < 60:
        return "взрослый 🌳"
    else:
        return "мудрец ✨"


async def photo_to_b64(update: Update) -> str:
    photo = update.message.photo[-1]
    file = await photo.get_file()
    raw = await file.download_as_bytearray()
    return base64.b64encode(raw).decode()


# ─── Хэндлеры ────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    db_user = await get_user(user.id)

    # Вернуть активного пользователя сразу в режим ACTIVE
    if db_user and db_user.get("state") == "active":
        pet = PETS[db_user["pet_type"]]
        await update.message.reply_text(
            f"{pet['emoji']} {pet['name']} соскучился!\n\n"
            "Присылай скрин с сахаром — жду 📸\n\n"
            "_/stats — посмотреть статус питомца_\n"
            "_/newpet — выбрать другого питомца_",
            parse_mode="Markdown",
        )
        return ACTIVE

    await save_user(user.id, first_name=user.first_name or "хозяйка", state="new")

    keyboard = [
        [
            InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k)
            for k, p in list(PETS.items())[:3]
        ],
        [
            InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k)
            for k, p in list(PETS.items())[3:]
        ],
    ]
    await update.message.reply_text(
        "Привет! 🌿\n\n"
        "Я помогу тебе следить за сахаром в крови — через маленького питомца, "
        "которого ты будешь выхаживать.\n\n"
        "Его самочувствие отражает твои показатели. "
        "Хороший сахар — он радуется. Высокий — переживает. "
        "При этом он знает *твою* норму и радуется *твоему* прогрессу.\n\n"
        "*Выбери своего питомца:*",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )
    return CHOOSING_PET


async def pet_chosen(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    pet_type = query.data

    if pet_type not in PETS:
        return CHOOSING_PET

    pet = PETS[pet_type]
    await save_user(query.from_user.id, pet_type=pet_type, state="calibrating")

    await query.edit_message_text(
        f"{pet['emoji']} *{pet['name']}*\n\n"
        f"{pet['backstory']}\n\n"
        "─────────────────────\n\n"
        f"Прежде чем начать, *{pet['name']} хочет познакомиться* с твоими показателями.\n\n"
        "📊 Пришли мне скрин со *средним сахаром за последний месяц* — "
        "обычно это вкладка «Статистика», «Отчёт» или «Аналитика» в твоём приложении.\n\n"
        f"Так {pet['name']} поймёт твою отправную точку и сможет радоваться "
        "каждому прогрессу — даже самому небольшому!\n\n"
        "_Или введи средний сахар вручную:_ `/setbaseline 9.5`",
        parse_mode="Markdown",
    )
    return CALIBRATING


async def calibration_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)
    pet = PETS[db_user["pet_type"]]

    thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю на скрин...")

    try:
        img_b64 = await photo_to_b64(update)
        baseline = await extract_calibration(img_b64)
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        baseline = None

    if not baseline:
        await thinking.edit_text(
            f"{pet['emoji']} Не смог найти средний сахар на этом скрине 😔\n\n"
            "Попробуй другой скрин — например, вкладку «Статистика» "
            "или «Средний уровень глюкозы» в приложении.\n\n"
            "Или введи вручную: `/setbaseline 9.5`",
            parse_mode="Markdown",
        )
        return CALIBRATING

    first_name = db_user.get("first_name", "хозяйка")
    await save_user(user_id, baseline=round(baseline, 1), state="active")

    await thinking.edit_text(
        f"{pet['emoji']} Понял! Средний сахар: *{baseline:.1f} ммоль/л*\n\n"
        f"{pet['name']} запомнил твою отправную точку, {first_name}. "
        f"Теперь он будет радоваться каждому улучшению — пусть даже на полмиллимоля!\n\n"
        "─────────────────────\n\n"
        "📸 *Присылай скрин утром и вечером*\n\n"
        "_/stats — статус питомца_",
        parse_mode="Markdown",
    )
    return ACTIVE


async def cmd_setbaseline(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)

    if not db_user or not db_user.get("pet_type"):
        await update.message.reply_text("Сначала выбери питомца — напиши /start!")
        return ConversationHandler.END

    try:
        val = float(ctx.args[0].replace(",", "."))
        if not (2.0 <= val <= 30.0):
            raise ValueError
    except (IndexError, ValueError):
        await update.message.reply_text("Напиши так: `/setbaseline 9.5`", parse_mode="Markdown")
        state = db_user.get("state", "calibrating")
        return CALIBRATING if state == "calibrating" else ACTIVE

    pet = PETS[db_user["pet_type"]]
    await save_user(user_id, baseline=val, state="active")

    await update.message.reply_text(
        f"{pet['emoji']} Запомнил! Базовый сахар: *{val:.1f} ммоль/л*\n\n"
        f"Теперь присылай скрин утром и вечером. {pet['name']} ждёт! 📸",
        parse_mode="Markdown",
    )
    return ACTIVE


async def daily_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)

    if not db_user or not db_user.get("baseline"):
        await update.message.reply_text(
            "Сначала нужно задать базовый уровень!\n"
            "Напиши `/setbaseline 9.5` (своё среднее значение)",
            parse_mode="Markdown",
        )
        return ACTIVE

    pet = PETS[db_user["pet_type"]]
    thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю...")

    try:
        img_b64 = await photo_to_b64(update)
        reading_data = await extract_reading(img_b64)
    except Exception as e:
        logger.error(f"Reading error: {e}")
        reading_data = None

    if not reading_data:
        await thinking.edit_text(
            f"{pet['emoji']} Не смог разобрать скрин 🤔\n"
            "Попробуй другой — желательно, чтобы цифра была хорошо видна."
        )
        return ACTIVE

    # Обновление серии
    today = str(date.today())
    yesterday = str(date.today() - timedelta(days=1))
    streak = db_user.get("streak", 0)
    last_date = db_user.get("last_date", "")

    if last_date == yesterday:
        streak += 1
    elif last_date != today:
        streak = 1

    # Подсчёт хороших дней (ниже базовой линии)
    total_good = db_user.get("total_good", 0)
    delta = reading_data["value_mmol"] - db_user["baseline"]
    if delta < 0 and reading_data["value_mmol"] >= 3.9 and last_date != today:
        total_good += 1

    await save_user(user_id, streak=streak, last_date=today, total_good=total_good)

    # Генерация реакции питомца
    try:
        reaction = await generate_reaction(
            pet_type=db_user["pet_type"],
            baseline=db_user["baseline"],
            reading=reading_data["value_mmol"],
            trend=reading_data.get("trend", "unknown"),
            streak=streak,
            first_name=db_user.get("first_name", "хозяйка"),
        )
    except Exception as e:
        logger.error(f"Reaction generation error: {e}")
        reaction = f"{pet['emoji']} Вижу показатель! Спасибо, что прислала."

    trend_arrow = {"up": " ↗️", "down": " ↘️", "stable": " ➡️"}.get(
        reading_data.get("trend", ""), ""
    )

    text = f"*{reading_data['value_mmol']:.1f} ммоль/л*{trend_arrow}\n\n{reaction}"

    if streak >= 7 and last_date != today:
        text += f"\n\n🔥🔥 *{streak} дней подряд!* {pet['name']} в полном восторге от тебя!"
    elif streak >= 3 and last_date != today:
        text += f"\n\n🔥 *{streak} дня подряд* — это настоящий подвиг!"

    await thinking.edit_text(text, parse_mode="Markdown")
    return ACTIVE


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)

    if not db_user or not db_user.get("pet_type"):
        await update.message.reply_text("Начни с /start!")
        return ACTIVE

    pet = PETS[db_user["pet_type"]]
    baseline = db_user.get("baseline") or 0
    streak = db_user.get("streak", 0)
    total_good = db_user.get("total_good", 0)
    stage = pet_stage(total_good)

    await update.message.reply_text(
        f"{pet['emoji']} *{pet['name']}* — {stage}\n\n"
        f"📊 Базовый сахар: *{baseline:.1f} ммоль/л*\n"
        f"🔥 Серия: *{streak}* дн. подряд\n"
        f"✅ Хороших дней: *{total_good}*\n\n"
        f"_{pet['name']} очень рад, что ты о нём заботишься!_",
        parse_mode="Markdown",
    )
    return ACTIVE


async def cmd_newpet(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await save_user(user_id, pet_type=None, baseline=None, streak=0, total_good=0, state="new")

    keyboard = [
        [
            InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k)
            for k, p in list(PETS.items())[:3]
        ],
        [
            InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k)
            for k, p in list(PETS.items())[3:]
        ],
    ]
    await update.message.reply_text(
        "Выбери нового питомца:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return CHOOSING_PET


async def unknown_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)

    if not db_user or not db_user.get("pet_type"):
        await update.message.reply_text("Напиши /start чтобы начать!")
        return ConversationHandler.END

    pet = PETS.get(db_user["pet_type"], PETS["bear"])
    text = update.message.text

    msg = await claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        system="""Пользователь пишет боту мониторинга сахара. Верни ТОЛЬКО JSON:
- Число глюкозы ("8.5", "сахар 9", "8,2 утром"): {"intent":"reading","value":8.5}
- Установить базовый ("базовый 9.5", "среднее 9", "моя норма 10"): {"intent":"baseline","value":9.5}
- Всё остальное: {"intent":"chat"}""",
        messages=[{"role": "user", "content": text}],
    )

    try:
        raw = "".join(b.text for b in msg.content if hasattr(b, "text"))
        data = json.loads(raw.replace("```json","").replace("```","").strip())
    except Exception:
        data = {"intent": "chat"}

    if data["intent"] == "reading" and db_user.get("baseline"):
        val = float(data["value"])
        thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю...")
        today = str(date.today())
        yesterday = str(date.today() - timedelta(days=1))
        streak = db_user.get("streak", 0)
        last_date = db_user.get("last_date", "")
        if last_date == yesterday:
            streak += 1
        elif last_date != today:
            streak = 1
        total_good = db_user.get("total_good", 0)
        if val < db_user["baseline"] and val >= 3.9 and last_date != today:
            total_good += 1
        await save_user(user_id, streak=streak, last_date=today, total_good=total_good)
        reaction = await generate_reaction(
            db_user["pet_type"], db_user["baseline"], val,
            "unknown", streak, db_user.get("first_name", "хозяйка"),
        )
        t = f"*{val:.1f} ммоль/л*\n\n{reaction}"
        if streak >= 3 and last_date != today:
            t += f"\n\n🔥 *{streak} дня подряд!*"
        await thinking.edit_text(t, parse_mode="Markdown")

    elif data["intent"] == "baseline":
        val = float(data["value"])
        await save_user(user_id, baseline=round(val, 1), state="active")
        await update.message.reply_text(
            f"{pet['emoji']} Запомнил! Базовый сахар: *{val:.1f} ммоль/л*\n\nТеперь присылай скрины или просто пиши цифру 📸",
            parse_mode="Markdown",
        )

    else:
        chat_msg = await claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=180,
            system=f"""Ты — {pet['name']} ({pet['emoji']}), виртуальный питомец.
Характер: {pet['personality']}
Отвечай на сообщение хозяйки. 2-3 предложения, с эмодзи, в своём характере.""",
            messages=[{"role": "user", "content": text}],
        )
        reply = "".join(b.text for b in chat_msg.content if hasattr(b, "text"))
        await update.message.reply_text(reply)

    return ACTIVE


# ─── Запуск ───────────────────────────────────────────────────────────────────

async def main():
    await init_db()

    persistence = PicklePersistence(filepath="glucopet_states.pkl")
    app = Application.builder().token(BOT_TOKEN).persistence(persistence).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            CHOOSING_PET: [
                CallbackQueryHandler(pet_chosen),
                CommandHandler("start", cmd_start),
            ],
            CALIBRATING: [
                MessageHandler(filters.PHOTO, calibration_photo),
                CommandHandler("setbaseline", cmd_setbaseline),
                CommandHandler("start", cmd_start),
            ],
            ACTIVE: [
                MessageHandler(filters.PHOTO, daily_photo),
                CommandHandler("stats", cmd_stats),
                CommandHandler("setbaseline", cmd_setbaseline),
                CommandHandler("newpet", cmd_newpet),
                MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_text),
            ],
        },
        fallbacks=[
            CommandHandler("start", cmd_start),
            CommandHandler("newpet", cmd_newpet),
        ],
        name="main_conv",
        persistent=True,
        allow_reentry=True,
    )

    app.add_handler(conv)

    logger.info("🐾 GlucoPet Bot запущен!")
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
