import os
import random
import torch
import clip
from PIL import Image
import telebot
from telebot import types

TOKEN = os.environ.get("BOT_TOKEN")
if not TOKEN:
    print("ERROR: BOT_TOKEN not set. Stop.")
    raise SystemExit("Set BOT_TOKEN environment variable before running.")

print("Starting bot...")

bot = telebot.TeleBot(TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model (this may take a minute)...")
model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP loaded on device:", device)

categories = {
    "paper": {
        "en": "📄 It's paper! Please recycle it in the paper bin ♻️. You can reuse for notes or crafts.",
        "ru": "📄 Это бумага! Сдайте в контейнер для бумаги ♻️. Можно использовать повторно.",
        "kk": "📄 Бұл қағаз! Қағаз қоқыс жәшігіне салыңыз ♻️. Қайта пайдалануға болады."
    },
    "plastic": {
        "en": "🍼 It's plastic! Rinse and recycle in the plastic bin. Reduce single-use plastics 🌱.",
        "ru": "🍼 Это пластик! Промойте и сдайте в контейнер для пластика. Меньше одноразового пластика 🌱.",
        "kk": "🍼 Бұл пластик! Жуып, пластик жәшігіне салыңыз. Бір реттік пластикті азайтыңыз 🌱."
    },
    "glass": {
        "en": "🍾 It's glass! Rinse bottles/jars and recycle. Wrap broken glass 🧤.",
        "ru": "🍾 Это стекло! Промыть бутылки/банки и сдать. Битое стекло упаковать 🧤.",
        "kk": "🍾 Бұл әйнек! Бөтелке/банкаларды жуып, қайта өңдеңіз. Сынғанын ораңыз 🧤."
    },
    "metal": {
        "en": "🥫 It's metal! Empty and rinse cans; recycle in metal bin 🔄.",
        "ru": "🥫 Это металл! Опорожните и промойте банки; сдайте на переработку 🔄.",
        "kk": "🥫 Бұл металл! Банкілерді босатып, жуып, қайта өңдеңіз 🔄."
    },
    "organic": {
        "en": "🍎 Organic waste — compost if possible. Keep it separate from plastics.",
        "ru": "🍎 Органика — компостируйте, если можно. Не смешивайте с пластиком.",
        "kk": "🍎 Органикалық қалдық — компостқа салыңыз. Пластикпен араластырмаңыз."
    },
    "e-waste": {
        "en": "💻 Electronic waste — take it to e-waste collection points. Never regular bins.",
        "ru": "💻 Электронные отходы — сдавайте в спец.пункты. Не в обычный мусор.",
        "kk": "💻 Электрондық қалдықтар — арнайы қабылдау пунктіне апарыңыз."
    },
    "batteries": {
        "en": "🔋 Batteries are hazardous — hand them in at special collection boxes.",
        "ru": "🔋 Батарейки — опасные. Сдавайте в специальные контейнеры.",
        "kk": "🔋 Батарейкалар қауіпті. Арнайы контейнерге тапсырыңыз."
    },
    "clothes": {
        "en": "👕 Clothes — donate if wearable, or recycle in textile bins.",
        "ru": "👕 Одежда — пожертвуйте или сдайтe для переработки.",
        "kk": "👕 Киім — қайырымдылыққа беріңіз немесе қайта өңдеңіз."
    },
    "cigarette": {
        "en": "🚬 Cigarette butt — throw only in a bin. Filters pollute soil and water.",
        "ru": "🚬 Окурок — выбрасывать только в урну. Фильтры загрязняют.",
        "kk": "🚬 Темекі тұқылы — тек қоқысқа тастаңыз. Фильтрлер ластайды."
    },
    "other": {
        "en": "♻️ Other — not a standard category, please dispose responsibly.",
        "ru": "♻️ Другое — не стандартная категория, утилизируйте ответственно.",
        "kk": "♻️ Басқа — стандартты санатқа жатпайды, жауапкершілікпен тастаңыз."
    }
}

prompts = [
    "a photo of paper",
    "a photo of a plastic bottle",
    "a photo of a glass bottle",
    "a photo of a metal can",
    "a photo of food scraps / organic waste",
    "a photo of electronic waste like a phone or laptop",
    "a photo of batteries",
    "a photo of clothing or textile",
    "a photo of a cigarette butt",
    "a photo of mixed rubbish / trash"
]

texts = clip.tokenize(prompts).to(device)

eco_facts = {
    "en": [
        "🌱 Recycling one glass bottle saves energy to power a computer ~25 minutes.",
        "🌳 Recycling one ton of paper saves about 17 trees.",
        "♻️ Recycling aluminum saves ~95% energy compared to new production."
    ],
    "ru": [
        "🌱 Переработка одной стеклянной бутылки экономит энергию для работы ПК ~25 минут.",
        "🌳 Переработка тонны бумаги сохраняет примерно 17 деревьев.",
        "♻️ Переработка алюминия экономит около 95% энергии."
    ],
    "kk": [
        "🌱 Бір әйнек бөтелкені қайта өңдеу компьютерді шамамен 25 минутқа қуаттандыруға жеткілікті энергияны үнемдейді.",
        "🌳 Бір тонна қағазды қайта өңдеу шамамен 17 ағашты сақтайды.",
        "♻️ Алюминийді қайта өңдеу жаңа өндіріске қарағанда ~95% энергия үнемдейді."
    ]
}

user_lang = {}

def get_lang(chat_id):
    return user_lang.get(chat_id, "en")

@bot.message_handler(commands=['start'])
def start_cmd(message):
    markup = types.InlineKeyboardMarkup()
    markup.row(types.InlineKeyboardButton("🌍 English", callback_data="lang_en"),
               types.InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
               types.InlineKeyboardButton("🇰🇿 Қазақша", callback_data="lang_kk"))
    bot.send_message(message.chat.id,
                     "💚 Hi eco-hero! I recognize 10 waste types: paper, plastic, glass, metal, organic, e-waste, batteries, clothes, cigarette, other.\n\nChoose your language:",
                     reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("lang_"))
def callback_lang(call):
    lang = call.data.split("_")[1]
    user_lang[call.message.chat.id] = lang
    bot.answer_callback_query(call.id, "Language set ✓")
    bot.send_message(call.message.chat.id, f"✅ Language set to {lang.upper()}.\nSend a photo or text (e.g. 'plastic bottle').")

@bot.message_handler(commands=['faq'])
def faq_cmd(message):
    lang = get_lang(message.chat.id)
    faq_text = {
        "en": "❓ FAQ:\n1) Send photo or text.\n2) I will suggest recycling instructions.\n3) Use /start to change language.",
        "ru": "❓ FAQ:\n1) Отправьте фото или текст.\n2) Я подскажу, как переработать.\n3) /start — смена языка.",
        "kk": "❓ FAQ:\n1) Сурет немесе мәтін жіберіңіз.\n2) Мен қайта өңдеу жөнінде айтамын.\n3) /start — тілді өзгерту."
    }
    bot.send_message(message.chat.id, faq_text[lang])

@bot.message_handler(content_types=['text'])
def handle_text(message):
    lang = get_lang(message.chat.id)
    low = message.text.strip().lower()
    for key in categories.keys():
        if key in low:
            reply = categories[key][lang]
            fact = random.choice(eco_facts[lang])
            bot.send_message(message.chat.id, f"{reply}\n\n🌍 Eco fact: {fact}")
            return
    bot.send_message(message.chat.id, {
        "en": "I didn't detect a category in text. Please send a photo or type the exact material (e.g. 'plastic').",
        "ru": "Не удалось определить категорию. Отправьте фото или напишите точнее (например 'plastic').",
        "kk": "Мәтінде санат анықталмады. Сурет жіберіңіз немесе нақты жазыңыз (мысалы 'plastic')."
    }[lang])

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id
    lang = get_lang(chat_id)
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded = bot.download_file(file_info.file_path)
    with open("temp.jpg", "wb") as f:
        f.write(downloaded)
    try:
        image = preprocess(Image.open("temp.jpg")).unsqueeze(0).to(device)
    except Exception as e:
        bot.send_message(chat_id, "Error processing image.")
        print("Image open error:", e)
        return
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits.cpu().numpy()[0]
    top_idx = int(probs.argmax())
    top_prob = float(probs[top_idx])
    label_key = list(categories.keys())[top_idx]
    reply_text = categories[label_key][lang]
    fact = random.choice(eco_facts[lang])
    if top_prob < 0.5:
        sorted_idx = probs.argsort()[::-1][:2]
        choice1 = list(categories.keys())[int(sorted_idx[0])]
        choice2 = list(categories.keys())[int(sorted_idx[1])]
        bot.send_message(chat_id,
                         f"🤔 I'm not fully sure. Looks like *{choice1}* ({probs[sorted_idx[0]]:.2f}) or *{choice2}* ({probs[sorted_idx[1]]:.2f}).\n"
                         f"If wrong, type the correct one.\n\nSuggested: {categories[choice1][lang]}\n\n🌍 Eco fact: {fact}",
                         parse_mode="Markdown")
    else:
        bot.send_message(chat_id, f"✅ I think *{label_key}*!\n\n{reply_text}\n\n🌍 Eco fact: {fact}", parse_mode="Markdown")
    print(f"[LOG] {chat_id} -> {label_key} (conf={top_prob:.2f})")

if __name__ == "__main__":
    print("Bot is polling... (press Ctrl+C to stop)")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)
