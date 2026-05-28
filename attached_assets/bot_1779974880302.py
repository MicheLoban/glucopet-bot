#!/usr/bin/env python3
import asyncio
import base64
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
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

MODEL = "claude-sonnet-4-6"

claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

# Regex for bare glucose numbers: "8.5", "8,2"
_GLUCOSE_RE = re.compile(r'^\s*(\d{1,2}[.,]\d)\s*$')


# ─────────────────────────────────────────
# Database
# ─────────────────────────────────────────

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id            INTEGER PRIMARY KEY,
                first_name         TEXT,
                pet_type           TEXT,
                baseline           REAL,
                streak             INTEGER DEFAULT 0,
                total_good         INTEGER DEFAULT 0,
                last_date          TEXT,
                state              TEXT DEFAULT 'new',
                reminder_morning   TEXT,
                reminder_evening   TEXT
            )
        """)
        # Readings history
        await db.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER,
                value        REAL,
                trend        TEXT,
                meal_context TEXT,
                ts           TEXT
            )
        """)
        # Migrate existing users table if reminder columns missing
        try:
            await db.execute("ALTER TABLE users ADD COLUMN reminder_morning TEXT")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN reminder_evening TEXT")
        except Exception:
            pass
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


async def save_reading(user_id: int, value: float, trend: str, meal_context: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO readings (user_id, value, trend, meal_context, ts) VALUES (?,?,?,?,?)",
            (user_id, value, trend, meal_context, datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_weekly_stats(user_id: int) -> Optional[dict]:
    """Returns avg, min, max, count for the last 7 days."""
    since = (datetime.utcnow() - timedelta(days=7)).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT AVG(value), MIN(value), MAX(value), COUNT(*) FROM readings WHERE user_id=? AND ts>=?",
            (user_id, since),
        ) as cur:
            row = await cur.fetchone()
            if row and row[3]:
                return {"avg": row[0], "min": row[1], "max": row[2], "count": row[3]}
    return None


async def get_all_users_with_reminders() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT user_id, pet_type, reminder_morning, reminder_evening FROM users "
            "WHERE state='active' AND (reminder_morning IS NOT NULL OR reminder_evening IS NOT NULL)"
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def meal_context_from_hour(hour: int) -> str:
    """Infer meal context from hour of day (UTC+0; good enough for reaction flavour)."""
    if 5 <= hour < 10:
        return "morning fasting"
    elif 10 <= hour < 13:
        return "post-breakfast"
    elif 13 <= hour < 16:
        return "post-lunch"
    elif 16 <= hour < 20:
        return "afternoon"
    elif 20 <= hour < 24:
        return "evening / post-dinner"
    else:
        return "night"


def pet_stage(total_good: int) -> str:
    if total_good < 5:
        return "малыш 🍼"
    elif total_good < 20:
        return "подросток 🌱"
    elif total_good < 60:
        return "взрослый 🌳"
    else:
        return "мудрец ✨"


def _try_parse_glucose_fast(text: str) -> Optional[float]:
    m = _GLUCOSE_RE.match(text.strip())
    if m:
        val = float(m.group(1).replace(",", "."))
        if 2.0 <= val <= 30.0:
            return val
    return None


# ─────────────────────────────────────────
# Claude helpers
# ─────────────────────────────────────────

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
        img_b64, media_type,
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
        img_b64, media_type,
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
        return data if data.get("found") else None
    except Exception as e:
        logger.error(f"extract_reading parse error: {e!r}")
        return None


async def analyze_dynamics(img_b64: str, media_type: str, pet: dict, baseline: float, first_name: str) -> str:
    system = (
        f"You are {pet['name']} ({pet['emoji']}), a virtual pet belonging to {first_name}, who has type 2 diabetes. "
        f"Character: {pet['personality_en']} "
        f"Her personal baseline average glucose is {baseline:.1f} mmol/L — compare everything to this. "
        "You are looking at a blood glucose chart or statistics screenshot. "
        "Describe what you see: general trend (rising, falling, stable), "
        "whether values are mostly above or below her personal baseline, any notable spikes or drops. "
        "Be warm, supportive, never judgmental. Never give medical or lifestyle advice. "
        "Never use *action in asterisks*. Use emojis. "
        "Write 3-4 sentences in first person, in character. Always respond in Russian."
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
    meal_context: str = "",
) -> str:
    pet = PETS[pet_type]
    delta = reading - baseline

    if reading < 3.9:
        situation = "ALERT: glucose very low (hypoglycemia), owner must eat something immediately"
        pet_mood = "scared and urgent"
    elif delta < -1.5:
        situation = "THRILLED: much better than owner's personal baseline, huge progress"
        pet_mood = "jumping with joy, absolutely delighted"
    elif delta < -0.5:
        situation = "HAPPY: better than owner's personal baseline"
        pet_mood = "very pleased and happy"
    elif delta <= 0.5:
        situation = "CALM: about the same as owner's personal baseline"
        pet_mood = "calm, quietly hopeful"
    elif delta <= 1.5:
        situation = "MILD SADNESS: slightly above owner's personal baseline"
        pet_mood = "a little sad but not judging, believes tomorrow will be better"
    else:
        situation = "UNWELL: significantly above owner's personal baseline"
        pet_mood = "worried, but does not judge or lecture"

    trend_note = {"up": "glucose is rising", "down": "glucose is dropping", "stable": "glucose is stable"}.get(trend, "")
    streak_note = (
        f"This is day {streak} in a row the owner sent readings — praise her consistency!"
        if streak >= 3 else ""
    )
    meal_note = f"Time of reading: {meal_context}." if meal_context else ""

    system_prompt = (
        f"You are {pet['name']} ({pet['emoji']}), a virtual pet. "
        f"Character: {pet['personality_en']} "
        f"Your owner {first_name} has type 2 diabetes. "
        f"Her PERSONAL baseline glucose: {baseline:.1f} mmol/L — always compare to this, never to generic medical ranges. "
        f"Today's reading: {reading:.1f} mmol/L ({delta:+.1f} from her baseline). {trend_note} "
        f"{meal_note} "
        f"Situation: {situation}. Your mood: {pet_mood}. "
        f"{streak_note} "
        "Write 2-3 sentences in first person, in character. "
        "Use emojis. NEVER judge, lecture, frighten, or give medical/lifestyle advice. "
        "Only love and care. Never use *action in asterisks*. "
        "Always respond in Russian."
    )

    msg = await claude.messages.create(
        model=MODEL,
        max_tokens=220,
        system=system_prompt,
        messages=[{"role": "user", "content": "React to the owner's glucose reading."}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


async def parse_text_intent(text: str) -> dict:
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


# ─────────────────────────────────────────
# Reminders (job_queue)
# ─────────────────────────────────────────

async def _send_reminder(context: ContextTypes.DEFAULT_TYPE):
    """Generic reminder job callback — data: {user_id, part}."""
    user_id = context.job.data["user_id"]
    part = context.job.data["part"]  # "morning" or "evening"

    db_user = await get_user(user_id)
    if not db_user or db_user.get("state") != "active":
        return

    pet = PETS.get(db_user["pet_type"], PETS["bear"])
    text = (
        f"{pet['emoji']} Доброе утро! Не забудь прислать утренний сахар 🌅"
        if part == "morning" else
        f"{pet['emoji']} Добрый вечер! Жду вечерний показатель 🌙"
    )
    try:
        await context.bot.send_message(chat_id=user_id, text=text)
    except Exception as e:
        logger.warning(f"Reminder send failed for {user_id}: {e!r}")


def _schedule_reminder(app: Application, user_id: int, time_str: str, part: str):
    """Schedule a daily reminder. time_str = 'HH:MM'."""
    try:
        h, m = map(int, time_str.split(":"))
        t = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).time()
        job_name = f"reminder_{user_id}_{part}"
        # Remove old job if exists
        current = app.job_queue.get_jobs_by_name(job_name)
        for job in current:
            job.schedule_removal()
        app.job_queue.run_daily(
            _send_reminder,
            time=t,
            name=job_name,
            data={"user_id": user_id, "part": part},
        )
        logger.info(f"Scheduled {part} reminder for user {user_id} at {time_str}")
    except Exception as e:
        logger.error(f"Failed to schedule reminder: {e!r}")


async def cmd_setreminder(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)

    if not db_user or db_user.get("state") != "active":
        await update.message.reply_text("Сначала настрой питомца — /start")
        return ACTIVE

    pet = PETS.get(db_user["pet_type"], PETS["bear"])
    args = ctx.args  # e.g. ["08:00"] or ["08:00", "21:00"] or ["off"]

    if not args or args[0].lower() == "off":
        await save_user(user_id, reminder_morning=None, reminder_evening=None)
        for part in ("morning", "evening"):
            for job in ctx.application.job_queue.get_jobs_by_name(f"reminder_{user_id}_{part}"):
                job.schedule_removal()
        await update.message.reply_text(f"{pet['emoji']} Напоминания отключены.")
        return ACTIVE

    time_re = re.compile(r'^\d{1,2}:\d{2}$')
    times = [a for a in args if time_re.match(a)]

    if not times:
        await update.message.reply_text(
            f"{pet['emoji']} Укажи время в формате ЧЧ:ММ\n\n"
            "Примеры:\n"
            "`/setreminder 08:00` — только утром\n"
            "`/setreminder 08:00 21:00` — утром и вечером\n"
            "`/setreminder off` — отключить",
            parse_mode="Markdown",
        )
        return ACTIVE

    morning = times[0] if len(times) >= 1 else None
    evening = times[1] if len(times) >= 2 else None

    await save_user(user_id, reminder_morning=morning, reminder_evening=evening)

    if morning:
        _schedule_reminder(ctx.application, user_id, morning, "morning")
    if evening:
        _schedule_reminder(ctx.application, user_id, evening, "evening")

    parts = []
    if morning:
        parts.append(f"утром в {morning}")
    if evening:
        parts.append(f"вечером в {evening}")

    await update.message.reply_text(
        f"{pet['emoji']} Буду напоминать {' и '.join(parts)}! 🔔\n\n"
        "_Отключить: /setreminder off_",
        parse_mode="Markdown",
    )
    return ACTIVE


# ─────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    db_user = await get_user(user.id)

    if db_user and db_user.get("state") == "active":
        pet = PETS[db_user["pet_type"]]
        await update.message.reply_text(
            f"{pet['emoji']} {pet['name']} скучал!\n\n"
            "Присылай скрин с сахаром — жду 📸\n\n"
            "_/stats · /setreminder · /newpet · /help_",
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
        "Его самочувствие отражает *твои* показатели относительно *твоей* нормы. "
        "Хороший сахар — он радуется. Высокий — переживает.\n\n"
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
        "📸 *Скрин с текущим сахаром* — пришли фото\n"
        "📊 *Скрин с графиком* — пришли и напиши «оцени динамику»\n"
        "🔢 *Просто напиши число* — `8.5` или `сахар 7.2`\n\n"
        "*Команды:*\n"
        "/stats — статистика питомца и недельный график\n"
        "/setbaseline 9.5 — задать базовый уровень вручную\n"
        "/setreminder 08:00 21:00 — напоминания (можно одно время)\n"
        "/setreminder off — отключить напоминания\n"
        "/newpet — сменить питомца\n"
        "/help — это сообщение",
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
        "📊 Пришли скрин со *средним сахаром за последний месяц* — "
        "вкладка «Статистика», «Отчёт» или «Аналитика» в твоём приложении.\n\n"
        "_Или введи вручную:_ `/setbaseline 9.5`",
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
        "Теперь он будет радоваться каждому улучшению — пусть даже на полмиллимоля!\n\n"
        "─────────────────────\n\n"
        "📸 *Присылай скрин утром и вечером*\n\n"
        "_/setreminder 08:00 21:00 — включи напоминания_\n"
        "_/stats · /help_",
        parse_mode="Markdown",
    )
    return ACTIVE


async def cmd_setbaseline(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)
    if not db_user or not db_user.get("pet_type"):
        await update.message.reply_text("Сначала выбери питомца — /start")
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
        f"Теперь присылай скрин. {pet['name']} ждёт! 📸",
        parse_mode="Markdown",
    )
    return ACTIVE


async def _process_reading(
    user_id: int, db_user: dict, pet: dict,
    value: float, trend: str,
    ctx: ContextTypes.DEFAULT_TYPE,
) -> str:
    """Shared logic: update streak/stats, save reading, generate reaction. Returns reply text."""
    today = str(date.today())
    yesterday = str(date.today() - timedelta(days=1))
    streak = db_user.get("streak", 0)
    last_date = db_user.get("last_date", "")

    if last_date == yesterday:
        streak += 1
    elif last_date != today:
        streak = 1

    total_good = db_user.get("total_good", 0)
    delta = value - db_user["baseline"]
    if delta < 0 and value >= 3.9 and last_date != today:
        total_good += 1

    meal_context = meal_context_from_hour(datetime.utcnow().hour)
    await save_user(user_id, streak=streak, last_date=today, total_good=total_good)
    await save_reading(user_id, value, trend, meal_context)
    ctx.user_data["last_reading"] = value

    reaction = await generate_reaction(
        pet_type=db_user["pet_type"],
        baseline=db_user["baseline"],
        reading=value,
        trend=trend,
        streak=streak,
        first_name=db_user.get("first_name", "хозяйка"),
        meal_context=meal_context,
    )

    trend_arrow = {"up": " ↗️", "down": " ↘️", "stable": " ➡️"}.get(trend, "")
    text = f"*{value:.1f} ммоль/л*{trend_arrow}\n\n{reaction}"

    if streak >= 7 and last_date != today:
        text += f"\n\n🔥🔥 *{streak} дней подряд!* {pet['name']} в полном восторге!"
    elif streak >= 3 and last_date != today:
        text += f"\n\n🔥 *{streak} дня подряд* — это настоящий подвиг!"

    return text


async def daily_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)

    if not db_user or not db_user.get("baseline"):
        await update.message.reply_text(
            "Сначала нужно задать базовый уровень!\n`/setbaseline 9.5`",
            parse_mode="Markdown",
        )
        return ACTIVE

    pet = PETS[db_user["pet_type"]]
    caption = (update.message.caption or "").lower()
    wants_dynamics = any(kw in caption for kw in ["динамик", "график", "тренд", "оцени", "посмотри"])
    thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю...")

    try:
        img_b64, media_type = await photo_to_b64(update)
    except Exception as e:
        logger.error(f"Photo download error: {e!r}", exc_info=True)
        await thinking.edit_text(f"{pet['emoji']} Не смог загрузить фото, попробуй ещё раз.")
        return ACTIVE

    ctx.user_data["last_photo"] = {"b64": img_b64, "media_type": media_type}

    if wants_dynamics:
        try:
            reply = await analyze_dynamics(img_b64, media_type, pet, db_user["baseline"], db_user.get("first_name", "хозяйка"))
        except Exception as e:
            logger.error(f"Dynamics error: {e!r}", exc_info=True)
            reply = f"{pet['emoji']} Вижу график, но не смог его разобрать 🤔"
        await thinking.edit_text(reply)
        return ACTIVE

    try:
        reading_data = await extract_reading(img_b64, media_type)
    except Exception as e:
        logger.error(f"Reading error: {e!r}", exc_info=True)
        reading_data = None

    if not reading_data:
        await thinking.edit_text(
            f"{pet['emoji']} Не смог разобрать скрин 🤔\n"
            "Попробуй другой — цифра должна быть хорошо видна.\n\n"
            "_Или напиши число вручную: `8.5`_",
            parse_mode="Markdown",
        )
        return ACTIVE

    try:
        text = await _process_reading(
            user_id, db_user, pet,
            reading_data["value_mmol"],
            reading_data.get("trend", "unknown"),
            ctx,
        )
    except Exception as e:
        logger.error(f"Process reading error: {e!r}", exc_info=True)
        text = f"{pet['emoji']} Вижу показатель! Спасибо, что прислала."

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
    weekly = await get_weekly_stats(user_id)

    last_line = f"📍 Последний: *{last_reading:.1f} ммоль/л*\n" if last_reading else ""

    weekly_block = ""
    if weekly and weekly["count"] >= 2:
        weekly_block = (
            f"\n📈 *За 7 дней* ({weekly['count']} замеров):\n"
            f"  Среднее: *{weekly['avg']:.1f}* · Мин: *{weekly['min']:.1f}* · Макс: *{weekly['max']:.1f}*\n"
        )

    reminder_line = ""
    mr = db_user.get("reminder_morning")
    er = db_user.get("reminder_evening")
    if mr or er:
        parts = []
        if mr: parts.append(f"🌅 {mr}")
        if er: parts.append(f"🌙 {er}")
        reminder_line = f"\n🔔 Напоминания: {' · '.join(parts)}\n"

    await update.message.reply_text(
        f"{pet['emoji']} *{pet['name']}* — {stage}\n\n"
        f"📊 Базовый сахар: *{baseline:.1f} ммоль/л*\n"
        f"{last_line}"
        f"{weekly_block}"
        f"🔥 Серия: *{streak}* дн. подряд\n"
        f"✅ Хороших дней: *{total_good}*\n"
        f"{reminder_line}\n"
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

    # Reading
    if intent_data.get("intent") == "reading" and db_user.get("baseline"):
        val = float(intent_data["value"])
        thinking = await update.message.reply_text(f"{pet['emoji']} Записываю...")
        try:
            reply = await _process_reading(user_id, db_user, pet, val, "unknown", ctx)
        except Exception as e:
            logger.error(f"Process reading error: {e!r}")
            reply = f"{pet['emoji']} Вижу показатель!"
        await thinking.edit_text(reply, parse_mode="Markdown")

    # Baseline
    elif intent_data.get("intent") == "baseline":
        val = float(intent_data["value"])
        await save_user(user_id, baseline=round(val, 1), state="active")
        await update.message.reply_text(
            f"{pet['emoji']} Запомнил! Базовый сахар: *{val:.1f} ммоль/л*\n\n"
            "Теперь присылай скрины или просто пиши цифру 📸",
            parse_mode="Markdown",
        )

    # Dynamics
    elif intent_data.get("intent") == "dynamics":
        last_photo = ctx.user_data.get("last_photo")
        if last_photo and db_user.get("baseline"):
            thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю на график...")
            try:
                reply = await analyze_dynamics(
                    last_photo["b64"], last_photo["media_type"], pet,
                    db_user["baseline"], db_user.get("first_name", "хозяйка"),
                )
            except Exception as e:
                logger.error(f"Dynamics error: {e!r}")
                reply = f"{pet['emoji']} Что-то пошло не так, попробуй прислать график снова."
            await thinking.edit_text(reply)
        else:
            await update.message.reply_text(
                f"{pet['emoji']} Пришли мне скрин с графиком сахара — я его оценю! 📊"
            )

    # Chat
    else:
        try:
            last_reading = ctx.user_data.get("last_reading")
            context_note = (
                f"The owner's last glucose reading was {last_reading:.1f} mmol/L. "
                if last_reading else ""
            )
            baseline_note = (
                f"Her personal baseline is {db_user['baseline']:.1f} mmol/L — "
                "always refer to this when discussing her glucose, never to generic medical ranges. "
                if db_user.get("baseline") else ""
            )
            chat_msg = await claude.messages.create(
                model=MODEL,
                max_tokens=180,
                system=(
                    f"You are {pet['name']} ({pet['emoji']}), a virtual pet. "
                    f"Character: {pet['personality_en']} "
                    f"Your owner has type 2 diabetes and you help her track blood sugar. "
                    f"{baseline_note}{context_note}"
                    "Respond in 2-3 sentences with emojis, in character. "
                    "Never give medical or lifestyle advice — you are a pet, not a doctor. "
                    "Never use *action in asterisks*. Always respond in Russian."
                ),
                messages=[{"role": "user", "content": text}],
            )
            reply = "".join(b.text for b in chat_msg.content if hasattr(b, "text"))
        except Exception:
            reply = f"{pet['emoji']} Пришли мне скрин с сахаром! 📸"
        await update.message.reply_text(reply)

    return ACTIVE


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    asyncio.run(init_db())
    app = Application.builder().token(BOT_TOKEN).build()

    # Restore reminders for existing users on startup
    async def post_init(application: Application):
        users = await get_all_users_with_reminders()
        for u in users:
            if u["reminder_morning"]:
                _schedule_reminder(application, u["user_id"], u["reminder_morning"], "morning")
            if u["reminder_evening"]:
                _schedule_reminder(application, u["user_id"], u["reminder_evening"], "evening")
        logger.info(f"Restored reminders for {len(users)} users")

    app.post_init = post_init

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
                CommandHandler("setreminder", cmd_setreminder),
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
