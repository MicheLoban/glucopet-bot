#!/usr/bin/env python3
import asyncio
import base64
import json
import logging
import os
import re
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
    filters,
)

from pets import PETS

load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN     = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]
DB_PATH       = os.environ.get("DB_PATH", "glucopet.db")

# Conversation states
CHOOSING_PET, CALIBRATING, ACTIVE = range(3)

# Updated model
MODEL = "claude-sonnet-4-6"

claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

# Regex for bare glucose numbers: "8.5", "8,2", "8" (1–2 digits, optional decimal)
_GLUCOSE_RE = re.compile(r'^\s*(\d{1,2}[.,]\d)\s*$')


# --- Database ---

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


# --- Claude API ---

def _detect_media_type(raw: bytes) -> str:
    if raw[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if raw[:2] == b'\xff\xd8':
        return "image/jpeg"
    if raw[:4] == b'GIF8':
        return "image/gif"
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        return "image/webp"
    return "image/jpeg"


async def photo_to_b64(update: Update):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    raw = bytes(await file.download_as_bytearray())
    media_type = _detect_media_type(raw)
    b64 = base64.b64encode(raw).decode("ascii")
    logger.info(f"Photo: media_type={media_type}, b64_len={len(b64)}")
    return b64, media_type


async def _call_claude_vision(img_b64: str, media_type: str, system: str, prompt: str, max_tokens: int = 400) -> str:
    msg = await claude.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


async def extract_calibration(img_b64: str, media_type: str) -> Optional[float]:
    text = await _call_claude_vision(
        img_b64,
        media_type,
        system=(
            "Analyze a glucose monitoring app screenshot. "
            "Find the average glucose level for a period (average glucose, mean glucose, средний сахар, среднее значение). "
            "Return ONLY JSON without markdown: {\"found\":true,\"value_mmol\":9.4} "
            "If value is in mg/dL divide by 18. If no average found: {\"found\":false}"
        ),
        prompt="Find the average glucose level for the period.",
    )
    try:
        data = json.loads(text.replace("```json", "").replace("```", "").strip())
        return float(data["value_mmol"]) if data.get("found") else None
    except Exception:
        return None


async def extract_reading(img_b64: str, media_type: str) -> Optional[dict]:
    text = await _call_claude_vision(
        img_b64,
        media_type,
        system=(
            "Analyze a blood glucose monitoring screenshot. "
            "Find the current glucose reading (текущий сахар, уровень глюкозы). "
            "Return ONLY JSON without markdown: {\"found\":true,\"value_mmol\":8.2,\"trend\":\"up\"} "
            "Trend must be: up, down, stable, or unknown. "
            "If value is in mg/dL divide by 18. If no reading found: {\"found\":false}"
        ),
        prompt="Find the current blood glucose reading.",
    )
    logger.info(f"extract_reading raw: {text!r}")
    try:
        data = json.loads(text.replace("```json", "").replace("```", "").strip())
        logger.info(f"extract_reading parsed: {data}")
        return data if data.get("found") else None
    except Exception as e:
        logger.error(f"extract_reading parse error: {e!r} raw={text!r}")
        return None


async def analyze_dynamics(img_b64: str, media_type: str, pet: dict, baseline: float, first_name: str) -> str:
    """Analyze glucose chart dynamics from a screenshot and respond in pet character."""
    pet_name = pet["name"]
    pet_emoji = pet["emoji"]
    pet_personality = pet["personality_en"]

    system = (
        f"You are {pet_name} ({pet_emoji}), a virtual pet belonging to {first_name}, who has type 2 diabetes. "
        f"Character: {pet_personality} "
        f"Her baseline average glucose is {baseline:.1f} mmol/L. "
        "You are looking at a blood glucose chart or statistics screenshot. "
        "Describe what you see in the chart: general trend (rising, falling, stable), "
        "whether values are mostly above or below the baseline, any notable spikes or drops. "
        "Be warm, supportive, never judgmental. Use emojis. "
        "Write 3-4 sentences in first person, in character. "
        "Never use *action in asterisks*. Always respond in Russian."
    )
    return await _call_claude_vision(
        img_b64, media_type, system,
        prompt="Describe the glucose dynamics shown in this chart.",
        max_tokens=300,
    )


async def generate_reaction(
    pet_type: str,
    baseline: float,
    reading: float,
    trend: str,
    streak: int,
    first_name: str,
) -> str:
    pet = PETS[pet_type]
    delta = reading - baseline

    if reading < 3.9:
        situation = "ALERT: glucose very low (hypoglycemia), must eat immediately"
        pet_mood = "scared and urgent, asks owner to eat something right now"
    elif delta < -1.5:
        situation = "THRILLED: much better than usual, huge progress"
        pet_mood = "jumping with joy, absolutely delighted"
    elif delta < -0.5:
        situation = "HAPPY: better than usual, noticeable progress"
        pet_mood = "very pleased and happy"
    elif delta <= 0.5:
        situation = "CALM: about the same as usual"
        pet_mood = "calm, quietly hopeful"
    elif delta <= 1.5:
        situation = "MILD SADNESS: slightly above usual"
        pet_mood = "a little sad but not judging, believes tomorrow will be better"
    else:
        situation = "UNWELL: significantly above usual"
        pet_mood = "feels bad and worried, but does not judge or lecture"

    trend_note = {"up": "glucose is rising", "down": "glucose is dropping", "stable": "glucose is stable"}.get(trend, "")
    streak_note = (
        f"This is day {streak} in a row that the owner has sent readings — praise her consistency!"
        if streak >= 3 else ""
    )

    pet_name = pet["name"]
    pet_emoji = pet["emoji"]
    pet_personality = pet["personality_en"]

    system_prompt = (
        f"You are {pet_name} ({pet_emoji}), a virtual pet. "
        f"Character: {pet_personality} "
        f"Your owner has type 2 diabetes. "
        f"Her baseline glucose: {baseline:.1f} mmol/L. "
        f"Today reading: {reading:.1f} mmol/L. {trend_note} "
        f"Delta from baseline: {delta:+.1f} mmol/L. "
        f"Situation: {situation} "
        f"Your mood: {pet_mood} "
        f"{streak_note} "
        f"Write 2-3 sentences in first person, in character. "
        f"Use emojis. NEVER judge, lecture, or frighten. Only love and care. "
        f"Never use *action in asterisks* — express emotions through words and emojis only. "
        f"Always respond in Russian."
    )

    msg = await claude.messages.create(
        model=MODEL,
        max_tokens=220,
        system=system_prompt,
        messages=[{"role": "user", "content": "React to the owner's glucose reading."}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


def _try_parse_glucose_fast(text: str) -> Optional[float]:
    """Try to parse a bare glucose number without calling Claude. Returns None if ambiguous."""
    m = _GLUCOSE_RE.match(text.strip())
    if m:
        val = float(m.group(1).replace(",", "."))
        if 2.0 <= val <= 30.0:
            return val
    return None


async def parse_text_intent(text: str) -> dict:
    """
    First tries a cheap regex for plain numbers.
    Falls back to Claude only for ambiguous/natural language messages.
    """
    # Fast path — plain number like "8.5" or "8,2"
    fast_val = _try_parse_glucose_fast(text)
    if fast_val is not None:
        return {"intent": "reading", "value": fast_val}

    msg = await claude.messages.create(
        model=MODEL,
        max_tokens=100,
        system=(
            "This is a BLOOD GLUCOSE monitoring bot for a person with type 2 diabetes. "
            "'Сахар' always means BLOOD SUGAR (glucose), never food or candy. "
            "Return ONLY JSON without markdown. "
            "If the message contains a blood glucose value (e.g. '8.5', 'сахар 9', 'глюкоза 8,2', 'у меня 7.4'): "
            "{\"intent\":\"reading\",\"value\":8.5}. "
            "If the user wants to set their baseline/norm (e.g. 'моя норма 9.5', 'базовый 10', 'среднее 9'): "
            "{\"intent\":\"baseline\",\"value\":9.5}. "
            "If the user asks to evaluate dynamics, trend, or chart (e.g. 'оцени динамику', 'как мой график', 'посмотри тренд'): "
            "{\"intent\":\"dynamics\"}. "
            "Otherwise: {\"intent\":\"chat\"}."
        ),
        messages=[{"role": "user", "content": text}],
    )
    raw = "".join(b.text for b in msg.content if hasattr(b, "text"))
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception:
        return {"intent": "chat"}


# --- Helpers ---

def pet_stage(total_good: int) -> str:
    if total_good < 5:
        return "малыш 🍼"
    elif total_good < 20:
        return "подросток 🌱"
    elif total_good < 60:
        return "взрослый 🌳"
    else:
        return "мудрец ✨"


def _update_streak(db_user: dict) -> tuple[int, int]:
    """Returns (new_streak, new_total_good) without saving."""
    today = str(date.today())
    yesterday = str(date.today() - timedelta(days=1))
    streak = db_user.get("streak", 0)
    last_date = db_user.get("last_date", "")

    if last_date == yesterday:
        streak += 1
    elif last_date != today:
        streak = 1

    return streak, today, last_date


# --- Handlers ---

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    db_user = await get_user(user.id)

    if db_user and db_user.get("state") == "active":
        pet = PETS[db_user["pet_type"]]
        await update.message.reply_text(
            f"{pet['emoji']} {pet['name']} скучал!\n\n"
            "Присылай скрин с сахаром — жду 📸\n\n"
            "_/stats — посмотреть статус питомца_\n"
            "_/newpet — выбрать другого питомца_\n"
            "_/help — что умеет бот_",
            parse_mode="Markdown",
        )
        return ACTIVE

    await save_user(user.id, first_name=user.first_name or "хозяйка", state="new")

    keyboard = [
        [InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k) for k, p in list(PETS.items())[:3]],
        [InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k) for k, p in list(PETS.items())[3:]],
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


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)
    pet = PETS.get(db_user["pet_type"], PETS["bear"]) if db_user and db_user.get("pet_type") else {"emoji": "🌿", "name": "питомец"}

    await update.message.reply_text(
        f"{pet['emoji']} *Что я умею:*\n\n"
        "📸 *Скрин с текущим сахаром* — пришли фото, я отвечу реакцией питомца\n"
        "📊 *Скрин с динамикой/графиком* — пришли и напиши «оцени динамику»\n"
        "🔢 *Просто напиши число* — например `8.5` или `сахар 7.2`\n\n"
        "*Команды:*\n"
        "/stats — статус питомца и твоя статистика\n"
        "/setbaseline 9.5 — задать базовый уровень сахара вручную\n"
        "/newpet — выбрать другого питомца\n"
        "/help — это сообщение\n\n"
        "_Питомец знает твой базовый уровень и радуется каждому улучшению — даже на 0.5 ммоль/л!_",
        parse_mode="Markdown",
    )
    return ACTIVE if (db_user and db_user.get("state") == "active") else CHOOSING_PET


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
        img_b64, media_type = await photo_to_b64(update)
        baseline = await extract_calibration(img_b64, media_type)
    except Exception as e:
        logger.error(f"Calibration error: {e!r}")
        baseline = None

    if not baseline:
        await thinking.edit_text(
            f"{pet['emoji']} Не смог найти средний сахар на этом скрине 😔\n\n"
            "Попробуй вкладку «Статистика» или «Средний уровень глюкозы».\n\n"
            "Или введи вручную: `/setbaseline 9.5`",
            parse_mode="Markdown",
        )
        return CALIBRATING

    await save_user(user_id, baseline=round(baseline, 1), state="active")

    await thinking.edit_text(
        f"{pet['emoji']} Понял! Средний сахар: *{baseline:.1f} ммоль/л*\n\n"
        f"{pet['name']} запомнил твою отправную точку. "
        f"Теперь он будет радоваться каждому улучшению — пусть даже на полмиллимоля!\n\n"
        "─────────────────────\n\n"
        "📸 *Присылай скрин утром и вечером*\n\n"
        "_/stats — статус питомца · /help — помощь_",
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
        return CALIBRATING if db_user.get("state") == "calibrating" else ACTIVE

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
            "Напиши `/setbaseline 9.5`",
            parse_mode="Markdown",
        )
        return ACTIVE

    pet = PETS[db_user["pet_type"]]

    # Check if user also sent text asking for dynamics analysis
    caption = (update.message.caption or "").lower()
    wants_dynamics = any(kw in caption for kw in ["динамик", "график", "тренд", "оцени", "посмотри"])

    thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю...")

    try:
        img_b64, media_type = await photo_to_b64(update)
    except Exception as e:
        logger.error(f"Photo download error: {e!r}", exc_info=True)
        await thinking.edit_text(f"{pet['emoji']} Не смог загрузить фото, попробуй ещё раз.")
        return ACTIVE

    # Store last photo in session for follow-up "оцени динамику" text messages
    ctx.user_data["last_photo"] = {"b64": img_b64, "media_type": media_type}

    if wants_dynamics:
        try:
            reply = await analyze_dynamics(
                img_b64, media_type, pet,
                db_user["baseline"],
                db_user.get("first_name", "хозяйка"),
            )
        except Exception as e:
            logger.error(f"Dynamics error: {e!r}", exc_info=True)
            reply = f"{pet['emoji']} Вижу график, но не смог его разобрать 🤔"
        await thinking.edit_text(reply)
        return ACTIVE

    # Normal reading extraction
    try:
        reading_data = await extract_reading(img_b64, media_type)
    except Exception as e:
        logger.error(f"Reading error: {e!r}", exc_info=True)
        reading_data = None

    if not reading_data:
        await thinking.edit_text(
            f"{pet['emoji']} Не смог разобрать скрин 🤔\n"
            "Попробуй другой — желательно, чтобы цифра была хорошо видна.\n\n"
            "_Или напиши число вручную, например: `8.5`_",
            parse_mode="Markdown",
        )
        return ACTIVE

    # Save last reading to session memory
    ctx.user_data["last_reading"] = reading_data["value_mmol"]

    today = str(date.today())
    yesterday = str(date.today() - timedelta(days=1))
    streak = db_user.get("streak", 0)
    last_date = db_user.get("last_date", "")

    if last_date == yesterday:
        streak += 1
    elif last_date != today:
        streak = 1

    total_good = db_user.get("total_good", 0)
    delta = reading_data["value_mmol"] - db_user["baseline"]
    if delta < 0 and reading_data["value_mmol"] >= 3.9 and last_date != today:
        total_good += 1

    await save_user(user_id, streak=streak, last_date=today, total_good=total_good)

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
        logger.error(f"Reaction error: {e!r}", exc_info=True)
        reaction = f"{pet['emoji']} Вижу показатель! Спасибо, что прислала."

    trend_arrow = {"up": " ↗️", "down": " ↘️", "stable": " ➡️"}.get(reading_data.get("trend", ""), "")
    text = f"*{reading_data['value_mmol']:.1f} ммоль/л*{trend_arrow}\n\n{reaction}"

    if streak >= 7 and last_date != today:
        text += f"\n\n🔥🔥 *{streak} дней подряд!* {pet['name']} в полном восторге!"
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

    last_reading = ctx.user_data.get("last_reading")
    last_line = f"📍 Последний показатель: *{last_reading:.1f} ммоль/л*\n" if last_reading else ""

    await update.message.reply_text(
        f"{pet['emoji']} *{pet['name']}* — {stage}\n\n"
        f"📊 Базовый сахар: *{baseline:.1f} ммоль/л*\n"
        f"{last_line}"
        f"🔥 Серия: *{streak}* дн. подряд\n"
        f"✅ Хороших дней: *{total_good}*\n\n"
        f"_{pet['name']} очень рад, что ты о нём заботишься!_",
        parse_mode="Markdown",
    )
    return ACTIVE


async def cmd_newpet(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    await save_user(user_id, pet_type=None, baseline=None, streak=0, total_good=0, state="new")
    ctx.user_data.clear()

    keyboard = [
        [InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k) for k, p in list(PETS.items())[:3]],
        [InlineKeyboardButton(f"{p['emoji']} {p['name']}", callback_data=k) for k, p in list(PETS.items())[3:]],
    ]
    await update.message.reply_text("Выбери нового питомца:", reply_markup=InlineKeyboardMarkup(keyboard))
    return CHOOSING_PET


async def unknown_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)

    if not db_user or not db_user.get("pet_type"):
        await update.message.reply_text("Напиши /start чтобы начать!")
        return ConversationHandler.END

    pet = PETS.get(db_user["pet_type"], PETS["bear"])
    text = update.message.text

    try:
        intent_data = await parse_text_intent(text)
    except Exception:
        intent_data = {"intent": "chat"}

    # --- Reading intent ---
    if intent_data.get("intent") == "reading" and db_user.get("baseline"):
        val = float(intent_data["value"])
        thinking = await update.message.reply_text(f"{pet['emoji']} Записываю...")

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
        ctx.user_data["last_reading"] = val

        try:
            reaction = await generate_reaction(
                db_user["pet_type"], db_user["baseline"], val,
                "unknown", streak, db_user.get("first_name", "хозяйка"),
            )
        except Exception as e:
            logger.error(f"Reaction error: {e!r}")
            reaction = f"{pet['emoji']} Вижу показатель!"

        t = f"*{val:.1f} ммоль/л*\n\n{reaction}"
        if streak >= 3 and last_date != today:
            t += f"\n\n🔥 *{streak} дня подряд!*"
        await thinking.edit_text(t, parse_mode="Markdown")

    # --- Baseline intent ---
    elif intent_data.get("intent") == "baseline":
        val = float(intent_data["value"])
        await save_user(user_id, baseline=round(val, 1), state="active")
        await update.message.reply_text(
            f"{pet['emoji']} Запомнил! Базовый сахар: *{val:.1f} ммоль/л*\n\n"
            "Теперь присылай скрины или просто пиши цифру 📸",
            parse_mode="Markdown",
        )

    # --- Dynamics intent: use last photo from session if available ---
    elif intent_data.get("intent") == "dynamics":
        last_photo = ctx.user_data.get("last_photo")
        if last_photo and db_user.get("baseline"):
            thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю на график...")
            try:
                reply = await analyze_dynamics(
                    last_photo["b64"], last_photo["media_type"], pet,
                    db_user["baseline"],
                    db_user.get("first_name", "хозяйка"),
                )
            except Exception as e:
                logger.error(f"Dynamics error: {e!r}")
                reply = f"{pet['emoji']} Что-то пошло не так, попробуй прислать график снова."
            await thinking.edit_text(reply)
        else:
            await update.message.reply_text(
                f"{pet['emoji']} Пришли мне скрин с графиком сахара — я его оценю! 📊"
            )

    # --- Chat fallback ---
    else:
        try:
            pet_name = pet["name"]
            pet_emoji = pet["emoji"]
            pet_personality = pet["personality_en"]
            last_reading = ctx.user_data.get("last_reading")
            context_note = (
                f"The owner's last glucose reading was {last_reading:.1f} mmol/L. "
                if last_reading else ""
            )
            chat_msg = await claude.messages.create(
                model=MODEL,
                max_tokens=180,
                system=(
                    f"You are {pet_name} ({pet_emoji}), a virtual pet. "
                    f"Character: {pet_personality} "
                    f"Your owner has type 2 diabetes and you help her track blood sugar. "
                    f"{context_note}"
                    "Respond to your owner's message in 2-3 sentences with emojis, in character. "
                    "Never use *action in asterisks* — express emotions through words and emojis only. "
                    "Always respond in Russian."
                ),
                messages=[{"role": "user", "content": text}],
            )
            reply = "".join(b.text for b in chat_msg.content if hasattr(b, "text"))
        except Exception:
            reply = f"{pet['emoji']} Пришли мне скрин с сахаром! 📸"
        await update.message.reply_text(reply)

    return ACTIVE


# --- Main ---

def main():
    asyncio.run(init_db())
    app = Application.builder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            CHOOSING_PET: [
                CallbackQueryHandler(pet_chosen),
                CommandHandler("start", cmd_start),
                CommandHandler("help", cmd_help),
            ],
            CALIBRATING: [
                MessageHandler(filters.PHOTO, calibration_photo),
                CommandHandler("setbaseline", cmd_setbaseline),
                CommandHandler("start", cmd_start),
                CommandHandler("help", cmd_help),
            ],
            ACTIVE: [
                MessageHandler(filters.PHOTO, daily_photo),
                CommandHandler("stats", cmd_stats),
                CommandHandler("setbaseline", cmd_setbaseline),
                CommandHandler("newpet", cmd_newpet),
                CommandHandler("help", cmd_help),
                MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_text),
            ],
        },
        fallbacks=[
            CommandHandler("start", cmd_start),
            CommandHandler("newpet", cmd_newpet),
            CommandHandler("help", cmd_help),
        ],
        allow_reentry=True,
    )

    app.add_handler(conv)
    logger.info("GlucoPet Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
