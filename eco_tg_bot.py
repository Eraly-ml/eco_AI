import torch
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
import json
import nest_asyncio
import os
import pickle
from fastai.vision.all import *
from PIL import Image
import gdown

url = "https://drive.google.com/uc?id=1aotofx1LuFD8-Pr76sOzez5CvPQaUT_k"
output = "my_eco_model.pkl"
gdown.download(url, output, quiet=False)


def load_model():
    try:
        # Загрузка модели
        model = load_learner('my_eco_model.pkl')
        print("Модель успешно загружена!")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None

# Классификация мусора через PyTorch с использованием FastAI
def classify_trash(learn, image_path):
    try:
        img = PILImage.create(image_path)  # Открытие изображения с помощью FastAI
        pred_class, pred_idx, outputs = learn.predict(img)  # Предсказание класса
        return str(pred_class)  # Возвращаем строковое представление класса
    except Exception as e:
        print(f"Ошибка при классификации изображения: {e}")
        return "Ошибка при классификации."


nest_asyncio.apply()

# Проверка наличия модели
if not os.path.exists('my_eco_model.pkl'):
    print("Файл модели не найден!")

# Команда /myid для вывода ID пользователя
async def myid(update: Update, context):
    user_id = update.message.from_user.id  # Получаем ID пользователя
    await update.message.reply_text(f"Ваш ID: {user_id}")
    
# Загрузка данных пользователей из JSON
def load_user_data():
    try:
        with open("user_data.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Сохранение данных пользователей в JSON
def save_user_data(user_data):
    with open("user_data.json", "w") as file:
        json.dump(user_data, file)

# Инициализация данных пользователей
user_data = load_user_data()

# Функция для начисления очков
def add_points(user_id, points):
    if user_id not in user_data:
        user_data[user_id] = {"points": 0}  # Если пользователя нет в данных, создаем запись с 0 очками
    user_data[user_id]["points"] += points  # Добавляем очки
    save_user_data(user_data)  # Сохраняем данные

# Команда /stat для получения очков пользователя по его user_id
async def stat(update: Update, context):
    # Проверка, что ID передан в команде
    if len(context.args) > 0:
        try:
            target_user_id = str(context.args[0])  # Преобразуем в строку
            user_data = load_user_data()  # Загружаем актуальные данные из файла
            print(f"Загруженные данные: {user_data}")  # Логируем данные


            # Проверяем, есть ли user_id в данных
            if target_user_id in user_data:
                points = user_data[target_user_id]["points"]
                await update.message.reply_text(f"Пользователь с ID {target_user_id} имеет {points} очков.")
            else:
                await update.message.reply_text(f"Пользователь с ID {target_user_id} не найден.")
        except ValueError:
            await update.message.reply_text("Пожалуйста, введите корректный числовой ID.")
    else:
        await update.message.reply_text("Пожалуйста, укажите ID пользователя после команды /stat.")


# Команда для получения топ-10 участников
async def top(update: Update, context):
    user_data = load_user_data()  # Загружаем актуальные данные перед выводом топа
    top_users = sorted(user_data.items(), key=lambda x: x[1]["points"], reverse=True)[:10]
    top_message = "Топ-10 участников по очкам:\n"
    for idx, (user_id, data) in enumerate(top_users, 1):
        top_message += f"{idx}. Пользователь {user_id}: {data['points']} очков\n"

    if not top_users:
        top_message = "Рейтинг пуст. Участвуйте в сборе мусора, чтобы попасть в топ!"

    await update.message.reply_text(top_message)

# Команда /my для вывода очков пользователя
async def my(update: Update, context):
    if update.message and update.message.from_user:
        user_id = update.message.from_user.id
        print(f"Получен user_id: {user_id}")  # Логируем user_id

        user_data = load_user_data()  # Загружаем актуальные данные перед отправкой
        if user_id in user_data:
            points = user_data[user_id]["points"]
            await update.message.reply_text(f"У вас {points} очков.")
        else:
            await update.message.reply_text("Вы еще не начали участвовать. Отправьте геолокацию и начните собирать мусор!")
    else:
        print("Ошибка: Не удалось получить user_id или update.message отсутствует.")
        await update.message.reply_text("Произошла ошибка: не удалось определить ваш идентификатор.")


# Стартовая команда
async def start(update: Update, context):
    user_id = update.message.from_user.id
    user_data = load_user_data()  # Загружаем актуальные данные
    if user_id not in user_data:
        user_data[user_id] = {"points": 0}  # Если пользователя нет в данных, создаем запись с 0 очками
        save_user_data(user_data)  # Сохраняем изменения
    await update.message.reply_text("Привет! Отправьте геолокацию места, где вы нашли мусор.")

# Обработка геолокации
async def location_handler(update: Update, context):
    user_id = update.message.from_user.id
    latitude = update.message.location.latitude
    longitude = update.message.location.longitude
    context.user_data['latitude'] = latitude
    context.user_data['longitude'] = longitude
    await update.message.reply_text("Геолокация сохранена! Теперь отправьте фото мусора.")

# Обработка фото мусора
async def photo_handler(update: Update, context):
    user_id = update.message.from_user.id
    latitude = context.user_data.get('latitude')
    longitude = context.user_data.get('longitude')
    photo = await update.message.photo[-1].get_file()
    photo_path = f"{user_id}_photo.jpg"
    await photo.download_to_drive(photo_path)

    # Загрузка модели и меток классов
    model = load_model()
    if not model:
        await update.message.reply_text("Ошибка загрузки модели. Попробуйте позже.")
        return

    trash_type = classify_trash(model, photo_path)
    context.user_data['photo_path'] = photo_path
    context.user_data['trash_type'] = trash_type

    await update.message.reply_text(f"Фото получено! Мусор классифицирован как: {trash_type}. Теперь отправьте видео, где вы выбрасываете этот мусор в контейнер.")

# Обработка видео
async def video_handler(update: Update, context):
    user_id = update.message.from_user.id
    video = await update.message.video.get_file()
    video_path = f"{user_id}_video.mp4"
    await video.download_to_drive(video_path)

    # Сохранение заявки
    latitude = context.user_data.get('latitude')
    longitude = context.user_data.get('longitude')
    photo_path = context.user_data.get('photo_path')
    trash_type = context.user_data.get('trash_type')

    # Отправка данных админу
    admin_chat_id = "-4797993721"  # Замените на ID чата администрации
    keyboard = [
        [InlineKeyboardButton("Принять", callback_data=f"approve_{user_id}"),
         InlineKeyboardButton("Отклонить", callback_data=f"reject_{user_id}")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Отправка сообщения с кнопками "Принять" и "Отклонить"
    await context.bot.send_message(chat_id=admin_chat_id, text=f"Заявка от пользователя {user_id}:\n"
                                                               f"Геолокация: {latitude}, {longitude}\n"
                                                               f"Тип мусора который отгадала ИИ: {trash_type}")
    await context.bot.send_photo(chat_id=admin_chat_id, photo=open(photo_path, 'rb'))
    await context.bot.send_video(chat_id=admin_chat_id, video=open(video_path, 'rb'), reply_markup=reply_markup)

    await update.message.reply_text("Спасибо! Ваша заявка отправлена на проверку.Проверьте свои очки с помощью команды /stat")

# Обработчик нажатий кнопок
async def button_handler(update: Update, context):
    query = update.callback_query
    await query.answer()

    action, user_id = query.data.split("_")

    if action == "approve":
        # Начисление очков после принятия заявки
        add_points(int(user_id), 5)  # Например, начисляем 5 очков за принятие заявки
        message = "Заявка принята! Вы получили 5 очков."
        admin_message = f"Заявка от пользователя {user_id} принята. Ему начислено 5 очков."
    elif action == "reject":
        message = "Заявка отклонена."
        admin_message = f"Заявка от пользователя {user_id} отклонена."

    # Отправляем сообщение админу в отдельное сообщение
    admin_chat_id = "-4797993721"  # Замените на ID чата администрации
    await context.bot.send_message(chat_id=admin_chat_id, text=admin_message)

    # Сохраняем изменения в файл
    save_user_data(user_data)

# Запуск приложения
async def main():
    application = Application.builder().token("7558746932:AAHLvdcnuDF2qdbtSMHKk8-bzJJk5WYqXTo").build()

    # Обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("my", my))  # Добавляем команду /my
    application.add_handler(MessageHandler(filters.LOCATION, location_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.VIDEO, video_handler))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(CommandHandler("stat", stat))  # Добавляем команду /stat
    application.add_handler(CommandHandler("top", top))
    application.add_handler(CommandHandler("myid", myid)) # Включаем обработчик команды /myid


    # Запуск
    await application.run_polling()

# Запуск бота
if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

