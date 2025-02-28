#!/usr/bin/env python3
import os
import time
import logging
import sqlite3
import numpy as np
import nest_asyncio
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from tensorflow.lite.python.interpreter import Interpreter
import folium
from folium.plugins import MarkerCluster
from telegram.error import TimedOut

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Загрузка переменных окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

# Применение nest_asyncio (полезно для работы в средах типа Jupyter/Colab)
nest_asyncio.apply()

# Инициализация базы данных
def init_db():
    try:
        conn = sqlite3.connect('bot_database.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                points INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                latitude REAL,
                longitude REAL,
                status TEXT,  -- 'clean' или 'polluted'
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("База данных успешно инициализирована.")
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы данных: {e}")
        raise

# Словарь для маппинга классов мусора
CLASS_MAPPING = {
    1: "стекло",
    2: "бумага",
    3: "картон",
    4: "пластик",
    5: "металл",
    6: "прочие отходы"
}

# Добавление пользователя в базу данных
def add_user(user_id):
    conn = sqlite3.connect('bot_database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    conn.commit()
    conn.close()

# Добавление локации в базу данных
def add_location(user_id, latitude, longitude, status):
    try:
        conn = sqlite3.connect('bot_database.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO locations (user_id, latitude, longitude, status)
            VALUES (?, ?, ?, ?)
        ''', (user_id, latitude, longitude, status))
        conn.commit()
        conn.close()
        logger.info("Локация успешно добавлена в базу данных.")
    except Exception as e:
        logger.error(f"Ошибка при добавлении локации: {e}")
        raise

# Добавление очков пользователю
def add_points(user_id, points):
    conn = sqlite3.connect('bot_database.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET points = points + ? WHERE user_id = ?', (points, user_id))
    conn.commit()
    conn.close()

# Получение очков пользователя
def get_user_points(user_id):
    conn = sqlite3.connect('bot_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT points FROM users WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0

# Получение топ-10 пользователей
def get_top_users():
    conn = sqlite3.connect('bot_database.db')
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, points FROM users ORDER BY points DESC LIMIT 10')
    result = cursor.fetchall()
    conn.close()
    return result

# Загрузка модели TensorFlow Lite
def load_tflite_model():
    try:
        interpreter = Interpreter(model_path="/home/eraly/projects/ml_dl/model.tflite")
        interpreter.allocate_tensors()
        logger.info("Модель TensorFlow Lite успешно загружена!")
        return interpreter
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None

# Классификация мусора по изображению
def classify_trash(interpreter, image_path):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = Image.open(image_path).resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_idx = np.argmax(output_data)
        class_name = CLASS_MAPPING.get(class_idx, "неизвестный класс")
        return f"Мусор классифицирован как: {class_name}"
    except Exception as e:
        logger.error(f"Ошибка при классификации изображения: {e}")
        return "Ошибка при классификации."

# Генерация карты загрязненных мест
def generate_pollution_map():
    conn = sqlite3.connect('bot_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT latitude, longitude FROM locations WHERE status = 'polluted'")
    polluted_locations = cursor.fetchall()
    conn.close()

    if polluted_locations:
        avg_latitude = sum(loc[0] for loc in polluted_locations) / len(polluted_locations)
        avg_longitude = sum(loc[1] for loc in polluted_locations) / len(polluted_locations)
        pollution_map = folium.Map(location=[avg_latitude, avg_longitude], zoom_start=12)

        for lat, lon in polluted_locations:
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="trash")
            ).add_to(pollution_map)

        map_file = "pollution_map.html"
        pollution_map.save(map_file)
        return map_file
    else:
        return None

# Обработчики команд и сообщений Telegram-бота

async def top(update: Update, context):
    top_users = get_top_users()
    top_message = "Топ-10 участников по очкам:\n"
    for idx, (user_id, points) in enumerate(top_users, 1):
        top_message += f"{idx}. Пользователь {user_id}: {points} очков\n"

    if not top_users:
        top_message = "Рейтинг пуст. Участвуйте в сборе мусора, чтобы попасть в топ!"

    await update.message.reply_text(top_message)

async def my(update: Update, context):
    if update.message and update.message.from_user:
        user_id = update.message.from_user.id
        points = get_user_points(user_id)
        await update.message.reply_text(f"У вас {points} очков.")
    else:
        logger.error("Ошибка: Не удалось получить user_id или update.message отсутствует.")
        await update.message.reply_text("Произошла ошибка: не удалось определить ваш идентификатор.")

async def start(update: Update, context):
    user_id = update.message.from_user.id
    add_user(user_id)
    await update.message.reply_text("Привет! Отправьте геолокацию места, где вы нашли мусор.")

async def polluted(update: Update, context):
    user_id = update.message.from_user.id
    add_user(user_id)
    context.user_data['is_polluted'] = True  # Устанавливаем флаг для команды /polluted
    await update.message.reply_text("Вы хотите сообщить о загрязнении. Отправьте геолокацию места.")

async def location_handler(update: Update, context):
    try:
        user_id = update.message.from_user.id

        if not update.message.location:
            await update.message.reply_text("Пожалуйста, отправьте геолокацию.")
            return

        latitude = update.message.location.latitude
        longitude = update.message.location.longitude

        logger.info(f"Получена геолокация: {latitude}, {longitude} от пользователя {user_id}")

        context.user_data['latitude'] = latitude
        context.user_data['longitude'] = longitude

        if context.user_data.get('is_polluted', False):
            try:
                add_location(user_id, latitude, longitude, 'polluted')
                await update.message.reply_text("Загрязнение отмечено! Спасибо за вашу помощь!")
            except Exception as e:
                logger.error(f"Ошибка при добавлении в базу данных: {e}")
                await update.message.reply_text("Произошла ошибка при сохранении данных. Попробуйте ещё раз.")
            finally:
                context.user_data.clear()
        else:
            await update.message.reply_text("Геолокация сохранена! Теперь отправьте фото мусора.")
    except Exception as e:
        logger.error(f"Ошибка в location_handler: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при обработке геолокации. Попробуйте ещё раз.")

async def photo_handler(update: Update, context):
    try:
        user_id = update.message.from_user.id
        photo = await update.message.photo[-1].get_file()
        photo_path = f"{user_id}_{int(time.time())}_photo.jpg"
        await photo.download_to_drive(photo_path)

        interpreter = context.application.context_data["interpreter"]
        if not interpreter:
            await update.message.reply_text("Ошибка загрузки модели. Попробуйте позже.")
            return

        trash_type = classify_trash(interpreter, photo_path)
        context.user_data['photo_path'] = photo_path
        context.user_data['trash_type'] = trash_type

        await update.message.reply_text(
            f"{trash_type}. Теперь отправьте видео, где вы выбрасываете этот мусор в контейнер."
        )
    except Exception as e:
        logger.error(f"Ошибка в photo_handler: {e}")
        await update.message.reply_text("Произошла ошибка при обработке фото. Попробуйте ещё раз.")

async def video_handler(update: Update, context):
    try:
        user_id = update.message.from_user.id

        if 'latitude' not in context.user_data or 'longitude' not in context.user_data:
            await update.message.reply_text("Сначала отправьте геолокацию.")
            return
        if 'photo_path' not in context.user_data or 'trash_type' not in context.user_data:
            await update.message.reply_text("Сначала отправьте фото мусора.")
            return

        latitude = context.user_data['latitude']
        longitude = context.user_data['longitude']
        photo_path = context.user_data['photo_path']
        trash_type = context.user_data['trash_type']

        video = await update.message.video.get_file()
        video_path = f"{user_id}_{int(time.time())}_video.mp4"
        await video.download_to_drive(video_path)

        if not os.path.exists(photo_path):
            logger.error(f"Фото не найдено: {photo_path}")
            await update.message.reply_text("Ошибка: фото не найдено.")
            return
        if not os.path.exists(video_path):
            logger.error(f"Видео не найдено: {video_path}")
            await update.message.reply_text("Ошибка: видео не найдено.")
            return

        keyboard = [
            [InlineKeyboardButton("Принять", callback_data=f"approve_{user_id}"),
             InlineKeyboardButton("Отклонить", callback_data=f"reject_{user_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=f"Заявка от пользователя {user_id}:\n"
                 f"Геолокация: {latitude}, {longitude}\n"
                 f"Тип мусора: {trash_type}"
        )
        with open(photo_path, 'rb') as photo_file:
            await context.bot.send_photo(chat_id=ADMIN_CHAT_ID, photo=photo_file)
        with open(video_path, 'rb') as video_file:
            await context.bot.send_video(chat_id=ADMIN_CHAT_ID, video=video_file, reply_markup=reply_markup)

        await update.message.reply_text("Спасибо! Ваша заявка отправлена на проверку.")
    except Exception as e:
        logger.error(f"Ошибка в video_handler: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при обработке видео. Попробуйте ещё раз.")
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)
        if os.path.exists(video_path):
            os.remove(video_path)
        context.user_data.clear()

async def map_command(update: Update, context):
    map_file = generate_pollution_map()
    if map_file:
        with open(map_file, 'rb') as f:
            await update.message.reply_document(
                document=f,
                caption="Карта загрязненных мест"
            )
    else:
        await update.message.reply_text("Загрязненные места не найдены.")

async def button_handler(update: Update, context):
    try:
        query = update.callback_query
        await query.answer()

        if "_" not in query.data:
            logger.error(f"Некорректный формат callback_data: {query.data}")
            return

        action, user_id = query.data.split("_")
        user_id = int(user_id)

        if action == "approve":
            add_points(user_id, 5)
            message = "Заявка принята! Вы получили 5 очков."
            admin_message = f"Заявка от пользователя {user_id} принята. Ему начислено 5 очков."
        elif action == "reject":
            message = "Заявка отклонена."
            admin_message = f"Заявка от пользователя {user_id} отклонена."
        else:
            logger.error(f"Неизвестное действие: {action}")
            return

        await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=admin_message)
        await query.message.reply_text(message)
    except Exception as e:
        logger.error(f"Ошибка в button_handler: {e}", exc_info=True)

async def main():
    init_db()

    application = Application.builder() \
        .token(TOKEN) \
        .read_timeout(30) \
        .write_timeout(30) \
        .pool_timeout(20) \
        .build()

    interpreter = load_tflite_model()
    if not interpreter:
        logger.error("Не удалось загрузить модель TensorFlow Lite. Бот остановлен.")
        return

    application.context_data = {"interpreter": interpreter}

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("polluted", polluted))
    application.add_handler(CommandHandler("my", my))
    application.add_handler(CommandHandler("top", top))
    application.add_handler(CommandHandler("map", map_command))
    application.add_handler(MessageHandler(filters.LOCATION, location_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.VIDEO, video_handler))
    application.add_handler(CallbackQueryHandler(button_handler))

    try:
        await application.run_polling()
    except TimedOut:
        logger.error("Произошёл тайм-аут. Перезапуск бота...")
        await main()
    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
