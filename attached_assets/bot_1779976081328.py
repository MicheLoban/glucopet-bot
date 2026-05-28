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

CHOOSING_PET, CALIBRATING, ACTIVE = range(3)
MODEL = "claude-sonnet-4-6"

claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)

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
        for col in ("reminder_morning", "reminder_evening"):
            try:
                await db.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
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


async def get_last_reading(user_id: int) -> Optional[dict]:
    """Returns the most recent reading for a user: {value, meal_context, ts, minutes_ago}."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT value, meal_context, ts FROM readings WHERE user_id=? ORDER BY ts DESC LIMIT 1",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            r = dict(row)
            try:
                dt = datetime.fromisoformat(r["ts"])
                r["minutes_ago"] = int((datetime.utcnow() - dt).total_seconds() / 60)
            except Exception:
                r["minutes_ago"] = None
            return r


async def get_weekly_stats(user_id: int) -> Optional[dict]:
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
            "Find the current glucose reading. "
            "Return ONLY JSON without markdown: {\"found\":true,\"value_mmol\":8.2,\"trend\":\"up\"} "
            "Trend: up, down, stable, or unknown. "
            "If value is in mg/dL divide by 18. If no reading found: {\"found\":false}"
        ),
        prompt="Find the current blood glucose reading.",
    )
    try:
        data = json.loads(text.replace("```json", "").replace("```", "").strip())
        return data if data.get("found") else None
    except Exception as e:
        logger.error(f"extract_reading error: {e!r}")
        return None


async def analyze_dynamics(img_b64: str, media_type: str, pet: dict, baseline: float, first_name: str) -> str:
    system = (
        f"You are {pet['name']} ({pet['emoji']}), a virtual pet belonging to {first_name}, who has type 2 diabetes. "
        f"Character: {pet['personality_en']} "
        f"Her personal baseline average glucose is {baseline:.1f} mmol/L. "
        "You are looking at a blood glucose chart or statistics screenshot. "
        "Describe the trend: rising, falling, or stable; whether values are above or below her baseline; "
        "any notable spikes or drops and when they occurred. "
        "Be warm, supportive, never judgmental. Never give medical or lifestyle advice. "
        "Never use *action in asterisks*. Use emojis. "
        "Write 3-4 sentences in first person. Always respond in Russian."
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
    user_context: str = "",
    prev_reading: Optional[float] = None,
    prev_minutes_ago: Optional[int] = None,
) -> str:
    pet = PETS[pet_type]
    delta_baseline = reading - baseline

    # ── Situation vs baseline ──────────────────────────────────────
    if reading < 3.9:
        vs_baseline = "ALERT: glucose very low (hypoglycemia) — owner must eat something immediately"
        base_mood = "scared and urgent"
    elif delta_baseline < -1.5:
        vs_baseline = "GREAT: much better than her personal baseline"
        base_mood = "jumping with joy"
    elif delta_baseline < -0.5:
        vs_baseline = "GOOD: better than her personal baseline"
        base_mood = "happy and pleased"
    elif delta_baseline <= 0.5:
        vs_baseline = "NEUTRAL: about the same as her personal baseline"
        base_mood = "calm, quietly hopeful"
    elif delta_baseline <= 1.5:
        vs_baseline = "SLIGHTLY HIGH: a bit above her personal baseline"
        base_mood = "a little worried but not judging"
    else:
        vs_baseline = "HIGH: significantly above her personal baseline"
        base_mood = "worried but still loving, does not lecture"

    # ── Situation vs previous reading ─────────────────────────────
    prev_block = ""
    if prev_reading is not None:
        delta_prev = reading - prev_reading
        time_note = f" ({prev_minutes_ago} min ago)" if prev_minutes_ago else ""
        if delta_prev <= -2.0:
            vs_prev = f"FALLING FAST: dropped {abs(delta_prev):.1f} mmol from previous reading{time_note} — great improvement right now"
        elif delta_prev <= -0.5:
            vs_prev = f"FALLING: down {abs(delta_prev):.1f} mmol from previous{time_note} — moving in the right direction"
        elif delta_prev < 0.5:
            vs_prev = f"STABLE: barely changed from previous{time_note}"
        elif delta_prev < 2.0:
            vs_prev = f"RISING: up {delta_prev:.1f} mmol from previous reading{time_note}"
        else:
            vs_prev = f"SPIKING: jumped {delta_prev:.1f} mmol from previous reading{time_note} — notable rise"
        prev_block = f"Change since last reading: {vs_prev}. Previous value: {prev_reading:.1f} mmol/L."

    # ── Context clues ──────────────────────────────────────────────
    context_note = ""
    if user_context:
        context_note = f"Owner says the context is: '{user_context}'. "
    elif meal_context:
        context_note = f"Time of day context: {meal_context}. "

    # ── Post-meal expectation calibration ─────────────────────────
    post_meal_note = ""
    is_post_meal = any(kw in (user_context + meal_context).lower()
                       for kw in ["lunch", "breakfast", "dinner", "meal", "обед", "завтрак", "ужин", "еда", "post"])
    if is_post_meal and delta_baseline > 0:
        post_meal_note = (
            "Note: a temporary post-meal rise is normal and expected. "
            "React more to whether it's coming down (good) or still rising (concerning), "
            "not just to the absolute value. "
        )

    streak_note = (
        f"This is day {streak} in a row — praise her consistency!" if streak >= 3 else ""
    )
    trend_note = {"up": "sensor shows rising", "down": "sensor shows falling", "stable": "sensor shows stable"}.get(trend, "")

    system_prompt = (
        f"You are {pet['name']} ({pet['emoji']}), a virtual pet. "
        f"Character: {pet['personality_en']} "
        f"Your owner {first_name} has type 2 diabetes. "
        f"Her PERSONAL baseline: {baseline:.1f} mmol/L. "
        f"Current reading: {reading:.1f} mmol/L ({delta_baseline:+.1f} from baseline). {trend_note}. "
        f"{prev_block} "
        f"{context_note}"
        f"{post_meal_note}"
        f"Overall situation vs baseline: {vs_baseline}. Your mood: {base_mood}. "
        f"{streak_note} "
        "IMPORTANT: React primarily to the DYNAMICS — is it getting better or worse compared to the last reading? "
        "Only mention the absolute value briefly. Focus on the direction and the story. "
        "Write 2-3 sentences in first person, in character. "
        "Use emojis. NEVER judge, lecture, frighten, or give medical/lifestyle advice. "
        "Never use *action in asterisks*. "
        "Always respond in Russian."
    )

    msg = await claude.messages.create(
        model=MODEL,
        max_tokens=240,
        system=system_prompt,
        messages=[{"role": "user", "content": "React to the owner's glucose reading."}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


async def parse_text_intent(text: str) -> dict:
    """
    Returns one of:
      {"intent":"reading", "value":8.5, "context":"после обеда"}
      {"intent":"baseline", "value":9.5}
      {"intent":"dynamics"}
      {"intent":"chat"}
    """
    fast_val = _try_parse_glucose_fast(text)
    if fast_val is not None:
        return {"intent": "reading", "value": fast_val, "context": ""}

    msg = await claude.messages.create(
        model=MODEL,
        max_tokens=120,
        system=(
            "This is a BLOOD GLUCOSE monitoring bot for a person with type 2 diabetes. "
            "'Сахар'/'сахарок' always means BLOOD SUGAR, never food. "
            "Return ONLY JSON without markdown. "

            "If the message contains a glucose value "
            "(e.g. '8.5', 'сахар 9', 'глюкоза 8,2', 'у меня 7.4', 'подлетел до 12', 'спустился до 10'): "
            "{\"intent\":\"reading\",\"value\":8.5,\"context\":\"после обеда\"} "
            "— extract any context the user mentions (после обеда/завтрака/ужина/сна/спорта/прогулки/натощак/перед сном). "
            "If no context mentioned, use empty string. "

            "If user wants to set baseline (моя норма 9.5, базовый 10, среднее 9): "
            "{\"intent\":\"baseline\",\"value\":9.5}. "

            "If user asks about dynamics/trend/chart (оцени динамику, как мой график, посмотри тренд): "
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
# Reminders
# ─────────────────────────────────────────

async def _send_reminder(context: ContextTypes.DEFAULT_TYPE):
    user_id = context.job.data["user_id"]
    part = context.job.data["part"]
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
        logger.warning(f"Reminder failed for {user_id}: {e!r}")


def _schedule_reminder(app: Application, user_id: int, time_str: str, part: str):
    try:
        h, m = map(int, time_str.split(":"))
        t = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0).time()
        job_name = f"reminder_{user_id}_{part}"
        for job in app.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()
        app.job_queue.run_daily(
            _send_reminder, time=t, name=job_name,
            data={"user_id": user_id, "part": part},
        )
    except Exception as e:
        logger.error(f"Schedule reminder error: {e!r}")


async def cmd_setreminder(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)
    if not db_user or db_user.get("state") != "active":
        await update.message.reply_text("Сначала настрой питомца — /start")
        return ACTIVE

    pet = PETS.get(db_user["pet_type"], PETS["bear"])
    args = ctx.args

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
    if morning: parts.append(f"утром в {morning}")
    if evening: parts.append(f"вечером в {evening}")
    await update.message.reply_text(
        f"{pet['emoji']} Буду напоминать {' и '.join(parts)}! 🔔\n\n"
        "_Отключить: /setreminder off_",
        parse_mode="Markdown",
    )
    return ACTIVE


# ─────────────────────────────────────────
# Core reading processor
# ─────────────────────────────────────────

async def _process_reading(
    user_id: int,
    db_user: dict,
    pet: dict,
    value: float,
    trend: str,
    ctx: ContextTypes.DEFAULT_TYPE,
    user_context: str = "",
) -> str:
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

    # Fetch previous reading for delta context
    prev_data = await get_last_reading(user_id)
    prev_reading = prev_data["value"] if prev_data else None
    prev_minutes_ago = prev_data.get("minutes_ago") if prev_data else None

    # Also check session (may be more recent than DB if not yet committed)
    session_prev = ctx.user_data.get("last_reading")
    if session_prev is not None and prev_reading is None:
        prev_reading = session_prev
        prev_minutes_ago = None

    await save_user(user_id, streak=streak, last_date=today, total_good=total_good)
    await save_reading(user_id, value, trend, user_context or meal_context)
    ctx.user_data["last_reading"] = value

    reaction = await generate_reaction(
        pet_type=db_user["pet_type"],
        baseline=db_user["baseline"],
        reading=value,
        trend=trend,
        streak=streak,
        first_name=db_user.get("first_name", "хозяйка"),
        meal_context=meal_context,
        user_context=user_context,
        prev_reading=prev_reading,
        prev_minutes_ago=prev_minutes_ago,
    )

    trend_arrow = {"up": " ↗️", "down": " ↘️", "stable": " ➡️"}.get(trend, "")
    text = f"*{value:.1f} ммоль/л*{trend_arrow}\n\n{reaction}"

    if streak >= 7 and last_date != today:
        text += f"\n\n🔥🔥 *{streak} дней подряд!* {pet['name']} в полном восторге!"
    elif streak >= 3 and last_date != today:
        text += f"\n\n🔥 *{streak} дня подряд* — настоящий подвиг!"

    return text


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
        "Я помогу тебе следить за сахаром через маленького питомца.\n\n"
        "Его самочувствие отражает *твои* показатели — и он видит динамику: "
        "растёт сейчас или падает, лучше чем час назад или хуже.\n\n"
        "*Выбери питомца:*",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="Markdown",
    )
    return CHOOSING_PET


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)
    pet = PETS.get(db_user["pet_type"], PETS["bear"]) if db_user and db_user.get("pet_type") else {"emoji": "🌿"}

    await update.message.reply_text(
        f"{pet['emoji']} *Что я умею:*\n\n"
        "📸 *Скрин с сахаром* — пришли фото\n"
        "📊 *Скрин с графиком* — пришли и напиши «оцени динамику»\n"
        "🔢 *Напиши число* — `8.5` или `после обеда 12` или `спустился до 10`\n\n"
        "Я вижу не только абсолютное значение, но и *динамику*: "
        "растёт ли сахар или падает относительно прошлого замера.\n\n"
        "*Команды:*\n"
        "/stats — статистика и недельный график\n"
        "/setbaseline 9.5 — задать базовый уровень\n"
        "/setreminder 08:00 21:00 — напоминания\n"
        "/setreminder off — отключить\n"
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
        f"*{pet['name']} хочет познакомиться* с твоими показателями.\n\n"
        "📊 Пришли скрин со *средним сахаром за последний месяц* — "
        "вкладка «Статистика» или «Аналитика» в твоём приложении.\n\n"
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
            f"{pet['emoji']} Не смог найти средний сахар 😔\n\n"
            "Попробуй вкладку «Статистика» или «Средний уровень глюкозы».\n\n"
            "Или введи вручную: `/setbaseline 9.5`",
            parse_mode="Markdown",
        )
        return CALIBRATING

    await save_user(user_id, baseline=round(baseline, 1), state="active")
    await thinking.edit_text(
        f"{pet['emoji']} Понял! Средний сахар: *{baseline:.1f} ммоль/л*\n\n"
        f"{pet['name']} запомнил твою отправную точку. "
        "Теперь он будет следить за каждым изменением!\n\n"
        "📸 *Присылай скрин утром и вечером*\n\n"
        "_/setreminder 08:00 21:00 — включи напоминания_",
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


async def daily_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    db_user = await get_user(user_id)
    if not db_user or not db_user.get("baseline"):
        await update.message.reply_text(
            "Сначала задай базовый уровень: `/setbaseline 9.5`",
            parse_mode="Markdown",
        )
        return ACTIVE

    pet = PETS[db_user["pet_type"]]
    caption = (update.message.caption or "").strip()
    caption_lower = caption.lower()
    wants_dynamics = any(kw in caption_lower for kw in ["динамик", "график", "тренд", "оцени", "посмотри"])
    thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю...")

    try:
        img_b64, media_type = await photo_to_b64(update)
    except Exception as e:
        logger.error(f"Photo error: {e!r}", exc_info=True)
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
            "Попробуй другой или напиши цифру: `8.5`",
            parse_mode="Markdown",
        )
        return ACTIVE

    try:
        text = await _process_reading(
            user_id, db_user, pet,
            reading_data["value_mmol"],
            reading_data.get("trend", "unknown"),
            ctx,
            user_context=caption,
        )
    except Exception as e:
        logger.error(f"Process reading error: {e!r}", exc_info=True)
        text = f"{pet['emoji']} Вижу показатель! Спасибо."

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
    mr, er = db_user.get("reminder_morning"), db_user.get("reminder_evening")
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
        user_context = intent_data.get("context", "")
        thinking = await update.message.reply_text(f"{pet['emoji']} Записываю...")
        try:
            reply = await _process_reading(user_id, db_user, pet, val, "unknown", ctx, user_context=user_context)
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
            "Теперь присылай скрины или пиши цифру 📸",
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
                reply = f"{pet['emoji']} Попробуй прислать график снова."
            await thinking.edit_text(reply)
        else:
            await update.message.reply_text(
                f"{pet['emoji']} Пришли скрин с графиком — я его оценю! 📊"
            )

    # Chat
    else:
        try:
            last_reading = ctx.user_data.get("last_reading")
            prev_data = await get_last_reading(user_id)

            context_note = ""
            if prev_data and last_reading:
                delta = last_reading - (prev_data["value"] if prev_data["value"] != last_reading else last_reading)
                context_note = f"Her last glucose reading was {last_reading:.1f} mmol/L. "
            elif last_reading:
                context_note = f"Her last glucose reading was {last_reading:.1f} mmol/L. "

            baseline_note = (
                f"Her personal baseline is {db_user['baseline']:.1f} mmol/L — "
                "always reference this, never generic medical ranges. "
                if db_user.get("baseline") else ""
            )
            chat_msg = await claude.messages.create(
                model=MODEL,
                max_tokens=180,
                system=(
                    f"You are {pet['name']} ({pet['emoji']}), a virtual pet. "
                    f"Character: {pet['personality_en']} "
                    f"Your owner has type 2 diabetes. "
                    f"{baseline_note}{context_note}"
                    "Respond in 2-3 sentences with emojis, in character. "
                    "Never give medical or lifestyle advice. "
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
