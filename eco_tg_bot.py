import os
import time
import logging
import sqlite3
import numpy as np
import nest_asyncio
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from tflite_runtime.interpreter import Interpreter

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

# Применение nest_asyncio для работы в Jupyter/Colab
nest_asyncio.apply()

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('bot_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            points INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

# Добавление пользователя в базу данных
def add_user(user_id):
    conn = sqlite3.connect('bot_database.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
    conn.commit()
    conn.close()

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
        interpreter = Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        logger.info("Модель TensorFlow Lite успешно загружена!")
        return interpreter
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None

# Классификация мусора
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
        return f"Класс мусора: {class_idx}"
    except Exception as e:
        logger.error(f"Ошибка при классификации изображения: {e}")
        return "Ошибка при классификации."

# Команда /myid
async def myid(update: Update, context):
    user_id = update.message.from_user.id
    await update.message.reply_text(f"Ваш ID: {user_id}")

# Команда /stat
async def stat(update: Update, context):
    if len(context.args) > 0:
        try:
            target_user_id = int(context.args[0])
            points = get_user_points(target_user_id)
            await update.message.reply_text(f"Пользователь с ID {target_user_id} имеет {points} очков.")
        except ValueError:
            await update.message.reply_text("Пожалуйста, введите корректный числовой ID.")
    else:
        await update.message.reply_text("Пожалуйста, укажите ID пользователя после команды /stat.")

# Команда /top
async def top(update: Update, context):
    top_users = get_top_users()
    top_message = "Топ-10 участников по очкам:\n"
    for idx, (user_id, points) in enumerate(top_users, 1):
        top_message += f"{idx}. Пользователь {user_id}: {points} очков\n"

    if not top_users:
        top_message = "Рейтинг пуст. Участвуйте в сборе мусора, чтобы попасть в топ!"

    await update.message.reply_text(top_message)

# Команда /my
async def my(update: Update, context):
    if update.message and update.message.from_user:
        user_id = update.message.from_user.id
        points = get_user_points(user_id)
        await update.message.reply_text(f"У вас {points} очков.")
    else:
        logger.error("Ошибка: Не удалось получить user_id или update.message отсутствует.")
        await update.message.reply_text("Произошла ошибка: не удалось определить ваш идентификатор.")

# Команда /start
async def start(update: Update, context):
    user_id = update.message.from_user.id
    add_user(user_id)
    await update.message.reply_text("Привет! Отправьте геолокацию места, где вы нашли мусор.")

# Обработчик геолокации
async def location_handler(update: Update, context):
    try:
        user_id = update.message.from_user.id
        latitude = update.message.location.latitude
        longitude = update.message.location.longitude
        context.user_data['latitude'] = latitude
        context.user_data['longitude'] = longitude
        await update.message.reply_text("Геолокация сохранена! Теперь отправьте фото мусора.")
    except Exception as e:
        logger.error(f"Ошибка в location_handler: {e}")
        await update.message.reply_text("Произошла ошибка при обработке геолокации. Попробуйте ещё раз.")

# Обработчик фото
async def photo_handler(update: Update, context):
    try:
        user_id = update.message.from_user.id
        latitude = context.user_data.get('latitude')
        longitude = context.user_data.get('longitude')
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

        await update.message.reply_text(f"Фото получено! Мусор классифицирован как: {trash_type}. Теперь отправьте видео, где вы выбрасываете этот мусор в контейнер.")
    except Exception as e:
        logger.error(f"Ошибка в photo_handler: {e}")
        await update.message.reply_text("Произошла ошибка при обработке фото. Попробуйте ещё раз.")
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)

# Обработчик видео
async def video_handler(update: Update, context):
    try:
        user_id = update.message.from_user.id
        video = await update.message.video.get_file()
        video_path = f"{user_id}_{int(time.time())}_video.mp4"
        await video.download_to_drive(video_path)

        latitude = context.user_data.get('latitude')
        longitude = context.user_data.get('longitude')
        photo_path = context.user_data.get('photo_path')
        trash_type = context.user_data.get('trash_type')

        keyboard = [
            [InlineKeyboardButton("Принять", callback_data=f"approve_{user_id}"),
             InlineKeyboardButton("Отклонить", callback_data=f"reject_{user_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"Заявка от пользователя {user_id}:\n"
                                                                  f"Геолокация: {latitude}, {longitude}\n"
                                                                  f"Тип мусора: {trash_type}")
        with open(photo_path, 'rb') as photo_file:
            await context.bot.send_photo(chat_id=ADMIN_CHAT_ID, photo=photo_file)
        with open(video_path, 'rb') as video_file:
            await context.bot.send_video(chat_id=ADMIN_CHAT_ID, video=video_file, reply_markup=reply_markup)

        await update.message.reply_text("Спасибо! Ваша заявка отправлена на проверку.")
    except Exception as e:
        logger.error(f"Ошибка в video_handler: {e}")
        await update.message.reply_text("Произошла ошибка при обработке видео. Попробуйте ещё раз.")
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

# Обработчик кнопок
async def button_handler(update: Update, context):
    try:
        query = update.callback_query
        await query.answer()

        action, user_id = query.data.split("_")
        user_id = int(user_id)

        if action == "approve":
            add_points(user_id, 5)
            message = "Заявка принята! Вы получили 5 очков."
            admin_message = f"Заявка от пользователя {user_id} принята. Ему начислено 5 очков."
        elif action == "reject":
            message = "Заявка отклонена."
            admin_message = f"Заявка от пользователя {user_id} отклонена."

        await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=admin_message)
        await query.edit_message_text(text=message)
    except Exception as e:
        logger.error(f"Ошибка в button_handler: {e}")

# Основная функция
async def main():
    init_db()  # Инициализация базы данных
    application = Application.builder().token(TOKEN).build()
    interpreter = load_tflite_model()
    if not interpreter:
        logger.error("Не удалось загрузить модель TensorFlow Lite. Бот остановлен.")
        return

    application.context_data = {"interpreter": interpreter}

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("my", my))
    application.add_handler(CommandHandler("stat", stat))
    application.add_handler(CommandHandler("top", top))
    application.add_handler(CommandHandler("myid", myid))
    application.add_handler(MessageHandler(filters.LOCATION, location_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.VIDEO, video_handler))
    application.add_handler(CallbackQueryHandler(button_handler))

    await application.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
