import torch
import clip
from PIL import Image
import telebot
from telebot import types
import random

TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

bot = telebot.TeleBot(TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

categories = {
    "paper": {
        "en": "📄 It's paper! Please recycle it in the blue bin ♻️. You can also reuse it for notes or crafts 🖊️✂️.",
        "ru": "📄 Это бумага! Сдайте её в контейнер для бумаги ♻️. Также можно использовать повторно для заметок или поделок 🖊️✂️.",
        "kk": "📄 Бұл қағаз! Оны көк жәшікке салыңыз ♻️. Қағазды жазбалар немесе қолөнер үшін қайта пайдалануға болады 🖊️✂️."
    },
    "plastic": {
        "en": "🍼 It's plastic! Always recycle it in a yellow bin ♻️. Reduce single-use plastics and try reusable bottles 🌱.",
        "ru": "🍼 Это пластик! Сдайте его в жёлтый контейнер ♻️. Старайтесь меньше использовать одноразовый пластик 🌱.",
        "kk": "🍼 Бұл пластик! Оны сары жәшікке салыңыз ♻️. Бір реттік пластикті азайтып, қайта пайдаланылатын бөтелкелерді қолданыңыз 🌱."
    },
    "glass": {
        "en": "🍾 It's glass! Recycle carefully in the green bin ♻️. Broken glass should be wrapped before disposal 🧤.",
        "ru": "🍾 Это стекло! Сдайте его в зелёный контейнер ♻️. Битое стекло нужно упаковать перед утилизацией 🧤.",
        "kk": "🍾 Бұл әйнек! Оны жасыл жәшікке салыңыз ♻️. Сынған әйнекті тастар алдында ораңыз 🧤."
    },
    "metal": {
        "en": "🥫 It's metal! Please recycle it in the gray bin ♻️. Aluminum cans can be infinitely recycled 🔄.",
        "ru": "🥫 Это металл! Сдайте его в серый контейнер ♻️. Алюминиевые банки можно перерабатывать бесконечно 🔄.",
        "kk": "🥫 Бұл металл! Оны сұр жәшікке салыңыз ♻️. Алюминий банкілерін шексіз қайта өңдеуге болады 🔄."
    },
    "organic": {
        "en": "🍎 It's organic waste! Compost it if you can 🌱. Turning food scraps into soil is the best gift for nature 🌍.",
        "ru": "🍎 Это органика! Компостируйте её, если возможно 🌱. Превращая отходы в почву, вы делаете лучший подарок природе 🌍.",
        "kk": "🍎 Бұл органикалық қалдық! Мүмкін болса, компостқа салыңыз 🌱. Қалдықтарды топыраққа айналдыру – табиғатқа ең үлкен сый 🌍."
    },
    "e-waste": {
        "en": "💻 It's electronic waste! Bring it to a special e-waste collection point ⚡. Never throw electronics into regular bins 🚫.",
        "ru": "💻 Это электронные отходы! Сдайте их в специальные пункты ⚡. Никогда не выбрасывайте электронику в обычные контейнеры 🚫.",
        "kk": "💻 Бұл электронды қалдықтар! Оларды арнайы қабылдау пунктіне апарыңыз ⚡. Электрониканы қарапайым жәшікке тастамаңыз 🚫."
    },
    "batteries": {
        "en": "🔋 It's a battery! Dispose of it only in special boxes ♻️. One battery can pollute 400 liters of water 💧.",
        "ru": "🔋 Это батарейка! Сдайте её только в специальные контейнеры ♻️. Одна батарейка может загрязнить 400 литров воды 💧.",
        "kk": "🔋 Бұл батарея! Оны тек арнайы қорапқа салыңыз ♻️. Бір батарея 400 литр суды ластай алады 💧."
    },
    "clothes": {
        "en": "👕 It's clothing! Donate it if it's still good, or recycle in textile bins 👗. Fast fashion is hurting the planet 🌍.",
        "ru": "👕 Это одежда! Если она в хорошем состоянии – пожертвуйте её, или сдайте в контейнер для текстиля 👗. Fast fashion вредит планете 🌍.",
        "kk": "👕 Бұл киім! Егер жақсы жағдайда болса – қайырымдылыққа беріңіз немесе тоқыма жәшігіне салыңыз 👗. Fast fashion табиғатқа зиян 🌍."
    },
    "cigarette": {
        "en": "🚬 It's a cigarette butt! Always throw it in a bin 🚮. Cigarette filters pollute soil and water with toxins 💀.",
        "ru": "🚬 Это окурок! Выбрасывайте только в урну 🚮. Фильтры сигарет загрязняют почву и воду токсинами 💀.",
        "kk": "🚬 Бұл темекі тұқылы! Оны тек қоқыс жәшігіне тастаңыз 🚮. Темекі сүзгілері топырақты және суды улайды 💀."
    },
    "other": {
        "en": "♻️ This item doesn’t belong to standard categories. Please dispose of it responsibly 🌍.",
        "ru": "♻️ Этот предмет не относится к стандартным категориям. Утилизируйте его ответственно 🌍.",
        "kk": "♻️ Бұл зат стандартты санаттарға жатпайды. Оны жауапкершілікпен тастаңыз 🌍."
    }
}

eco_facts = {
    "en": [
        "🌱 Recycling one glass bottle saves enough energy to power a computer for 25 minutes.",
        "🌳 Every ton of recycled paper saves 17 trees.",
        "♻️ Plastic takes up to 500 years to decompose.",
        "💡 Recycling aluminum saves 95% of the energy needed to make new aluminum."
    ],
    "ru": [
        "🌱 Переработка одной стеклянной бутылки экономит энергию на работу компьютера 25 минут.",
        "🌳 Каждая тонна переработанной бумаги сохраняет 17 деревьев.",
        "♻️ Пластику нужно до 500 лет, чтобы разложиться.",
        "💡 Переработка алюминия экономит 95% энергии."
    ],
    "kk": [
        "🌱 Бір әйнек бөтелкені қайта өңдеу компьютерді 25 минутқа қуаттандыруға жеткілікті энергия үнемдейді.",
        "🌳 Әр тонна қайта өңделген қағаз 17 ағашты сақтайды.",
        "♻️ Пластиктің ыдырауына 500 жылға дейін уақыт кетеді.",
        "💡 Алюминийді қайта өңдеу энергияның 95% үнемдейді."
    ]
}

user_lang = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("🌍 English", callback_data="lang_en"))
    markup.add(types.InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"))
    markup.add(types.InlineKeyboardButton("🇰🇿 Қазақша", callback_data="lang_kk"))
    bot.send_message(
        message.chat.id,
        "💚 Hii, eco-hero! 🌍 Thank you for caring about recycling!\n\n"
        "✨ I can recognize 10 types of waste: 📄 paper, 🍼 plastic, 🍾 glass, 🥫 metal, 🍎 organic, 💻 e-waste, 🔋 batteries, 👕 clothes, 🚬 cigarette, ♻️ other.\n\n"
        "Please choose your language 🌐:",
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("lang_"))
def set_language(call):
    lang = call.data.split("_")[1]
    user_lang[call.message.chat.id] = lang
    bot.send_message(
        call.message.chat.id,
        f"✅ Language set: {lang.upper()}!\n\n"
        "📸 Send me a picture or 📝 type a waste name — I'll guide you on recycling ♻️✨"
    )

@bot.message_handler(commands=['faq'])
def faq(message):
    lang = user_lang.get(message.chat.id, "en")
    faqs = {
        "en": "❓ FAQ:\n1️⃣ Send me a photo of waste 📸\n2️⃣ Or type the waste name 📝\n3️⃣ I will tell you how to recycle it ♻️.",
        "ru": "❓ FAQ:\n1️⃣ Отправьте фото мусора 📸\n2️⃣ Или напишите его название 📝\n3️⃣ Я скажу, как его утилизировать ♻️.",
        "kk": "❓ FAQ:\n1️⃣ Маған қоқыс суретін жіберіңіз 📸\n2️⃣ Немесе атауын жазыңыз 📝\n3️⃣ Мен оны қалай өңдеу керегін айтамын ♻️."
    }
    bot.send_message(message.chat.id, faqs[lang])

@bot.message_handler(content_types=['text'])
def handle_text(message):
    lang = user_lang.get(message.chat.id, "en")
    text = message.text.lower()
    found = None
    for cat in categories.keys():
        if cat in text:
            found = cat
            break
    if found:
        reply = categories[found][lang]
    else:
        reply = categories["other"][lang]
    fact = random.choice(eco_facts[lang])
    bot.send_message(message.chat.id, f"{reply}\n\n🌍 Eco Fact: {fact}")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    lang = user_lang.get(message.chat.id, "en")
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded = bot.download_file(file_info.file_path)
    with open("temp.jpg", "wb") as f:
        f.write(downloaded)
    image = preprocess(Image.open("temp.jpg")).unsqueeze(0).to(device)
    texts = clip.tokenize(list(categories.keys())).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        logits_per_image, _ = model(image, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    idx = probs[0].argmax()
    pred = list(categories.keys())[idx]
    reply = categories[pred][lang]
    fact = random.choice(eco_facts[lang])
    bot.send_message(message.chat.id, f"{reply}\n\n🌍 Eco Fact: {fact}")

bot.polling()
