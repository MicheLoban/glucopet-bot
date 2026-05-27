#!/usr/bin/env python3
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

claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)


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


async def _call_claude_vision(img_b64: str, media_type: str, system: str, prompt: str) -> str:
    msg = await claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
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
        system="Analyze a glucose monitoring app screenshot. Find the average glucose level for a period (average glucose, mean glucose). Return ONLY JSON without markdown: {\"found\":true,\"value_mmol\":9.4} If value is in mg/dL divide by 18. If no average found: {\"found\":false}",
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
        system="Analyze a blood glucose monitoring screenshot. Find the current glucose reading. Return ONLY JSON without markdown: {\"found\":true,\"value_mmol\":8.2,\"trend\":\"up\"} Trend must be: up, down, stable, or unknown. If value is in mg/dL divide by 18. If no reading found: {\"found\":false}",
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
    streak_note = f"This is day {streak} in a row that the owner has sent readings - praise her consistency!" if streak >= 3 else ""

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
        f"Always respond in Russian."
    )

    msg = await claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=220,
        system=system_prompt,
        messages=[{"role": "user", "content": "React to the owner's glucose reading."}],
    )
    return "".join(b.text for b in msg.content if hasattr(b, "text"))


async def parse_text_intent(text: str) -> dict:
    msg = await claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        system="The user sends a message to a glucose monitoring bot. Return ONLY JSON without markdown. If the message contains a glucose number (like '8.5', 'sugar 9', '8,2'): {\"intent\":\"reading\",\"value\":8.5}. If they want to set baseline ('baseline 9.5', 'my norm 10', 'average 9'): {\"intent\":\"baseline\",\"value\":9.5}. Otherwise: {\"intent\":\"chat\"}",
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
        return "baby 🍼"
    elif total_good < 20:
        return "teenager 🌱"
    elif total_good < 60:
        return "adult 🌳"
    else:
        return "wise one ✨"


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
            "_/newpet — выбрать другого питомца_",
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
    thinking = await update.message.reply_text(f"{pet['emoji']} Смотрю...")

    try:
        img_b64, media_type = await photo_to_b64(update)
        reading_data = await extract_reading(img_b64, media_type)
    except Exception as e:
        logger.error(f"Reading error: {e!r}", exc_info=True)
        reading_data = None

    if not reading_data:
        await thinking.edit_text(
            f"{pet['emoji']} Не смог разобрать скрин 🤔\n"
            "Попробуй другой — желательно, чтобы цифра была хорошо видна."
        )
        return ACTIVE

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

    if intent_data.get("intent") == "reading" and db_user.get("baseline"):
        val = float(intent_data["value"])
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

    elif intent_data.get("intent") == "baseline":
        val = float(intent_data["value"])
        await save_user(user_id, baseline=round(val, 1), state="active")
        await update.message.reply_text(
            f"{pet['emoji']} Запомнил! Базовый сахар: *{val:.1f} ммоль/л*\n\nТеперь присылай скрины или просто пиши цифру 📸",
            parse_mode="Markdown",
        )

    else:
        try:
            pet_name = pet["name"]
            pet_emoji = pet["emoji"]
            pet_personality = pet["personality_en"]
            chat_msg = await claude.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=180,
                system=f"You are {pet_name} ({pet_emoji}), a virtual pet. Character: {pet_personality} Respond to your owner's message in 2-3 sentences with emojis, in character. Always respond in Russian.",
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
        allow_reentry=True,
    )

    app.add_handler(conv)
    logger.info("GlucoPet Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
