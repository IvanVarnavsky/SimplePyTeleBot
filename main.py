import random
import re
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

with open("./data/examples_bot_intents.json", "r") as intents_file:
    BIG_INTENTS = json.load(intents_file)

INTENTS = BIG_INTENTS["intents"]
x = []  # Фразы
y = []  # Интенты
failure_phrases = BIG_INTENTS['failure_phrases']

for name, intent in INTENTS.items():
    for phrase in intent['examples']:
        x.append(phrase)
        y.append(name)

    for phrase in intent['responses']:
        x.append(phrase)
        y.append(name)

vectorizer = CountVectorizer()
vectorizer.fit(x)
vecX = vectorizer.transform(x)

mlp_model = MLPClassifier(max_iter=500)
mlp_model.fit(vecX, y)

rf_model = RandomForestClassifier()
rf_model.fit(vecX, y)

# Выбираем модель которая лучше обучилась
if mlp_model.score(vecX, y) > rf_model.score(vecX, y):
    MODEL = mlp_model
else:
    MODEL = rf_model


def get_intent_ml(text):
    vec_text = vectorizer.transform([text])
    intent_ml = MODEL.predict(vec_text)[0]
    return intent_ml


# Функция очистки текста от знаков препинания и лишних пробелов, а так же приведение к нижнему регистру.
def filter_text(text):
    text = text.lower().strip()
    expression = r'[^\w\s]'
    return re.sub(expression, "", text)


# Функция сравнивает текст пользователя с примером и решает похожи ли они
def text_match(user_text, example):
    # Убираем все лишнее
    user_text = filter_text(user_text)
    example = filter_text(example)

    if len(user_text) == 0 or len(example) == 0:
        return False

    example_length = len(example)  # Длина фразы example
    if example_length == 0:
        return False
    # На сколько в % отличаются фразы
    difference = nltk.edit_distance(user_text, example) / example_length
    return difference < 0.2  # Если разница меньше 20%


# Определить намерение по тексту
def get_intent(text):
    # Проверить все существующие intent'ы
    for intent_name in INTENTS.keys():
        examples = INTENTS[intent_name]["examples"]
        # Проверить все examples
        for example in examples:
            # Какой-нибудь один будет иметь example похожий на text
            if text_match(text, example):
                return intent_name


# Берёт случайный response для данного intent'а
def get_response(intent):
    return random.choice(INTENTS[intent]["responses"])


# Сам бот
def bot(text):
    text = filter_text(text)
    intent = get_intent(text)  # Найти намерение
    if not intent:  # Если намерение не найдено
        intent = get_intent_ml(text)  # Пробуем подключить МЛ модель

    if intent:  # Если нашлось в итоге, выводим ответ
        return get_response(intent)
    else:
        return random.choice(failure_phrases)  # Выводим случайную заглушку из фраз ошибок


# Приветствие пользователя по команде /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.first_name}')


# Функция для MessageHandler'а, вызывать ее при каждом сообщении боту
async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text
    reply_t = bot(question)
    await update.message.reply_text(reply_t)  # Ответ пользователю


TOKEN = 'Your token'  # Get your token from BotFather
app = ApplicationBuilder().token(TOKEN).build()

# Добавляем обработчик /start
app.add_handler(CommandHandler("start", start))

# Создаем обработчик текстовых сообщений
handler = MessageHandler(filters.Text(), reply)
app.add_handler(handler)  # Добавляем обработчик в приложение

# Запускаем опрос Telegram
app.run_polling()
