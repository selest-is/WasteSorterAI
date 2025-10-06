WASTE SORTER AI BOT

WasteSorterAI is a smart Telegram bot that helps users identify and correctly recycle various types of waste using AI image recognition (OpenAI CLIP).
By simply sending a photo or typing the name of an item, users receive instant recycling guidance in English, Russian, or Kazakh.

FEATURES

AI-powered recognition: Uses CLIP (ViT-B/32) to identify waste categories from photos.

Multilingual support: English, Russian, and Kazakh.

10 waste categories: paper, plastic, glass, metal, organic, e-waste, batteries, clothes, cigarette, and other.

Eco tips and facts: Provides random environmental facts and recycling tips.

Simple interface: Works via Telegram with easy text and photo commands.

HOW IT WORKS

Start the bot with /start

Choose your language

Send a photo of waste or type its name (for example, “plastic bottle”)

The bot replies with:

The detected waste category

Recycling instructions

A random eco fact

EXAMPLE

User: (sends a picture of a plastic bottle)
Bot:
"It's plastic! Always recycle it in a yellow bin.
Reduce single-use plastics and try reusable bottles.

Eco Fact: Recycling aluminum saves 95% of the energy needed to make new aluminum."

INSTALLATION

Clone the repository:
git clone https://github.com/yourusername/WasteSorterAI.git

cd WasteSorterAI

Install dependencies:
pip install torch torchvision torchaudio telebot pillow git+https://github.com/openai/CLIP.git

Set your Telegram Bot Token:
Open the file WasteSorterAI.py and replace the line
TOKEN = "YOUR_BOT_TOKEN"
with your actual bot token from @BotFather.

Run the bot:
python3 WasteSorterAI.py

REQUIREMENTS

Python 3.9 or newer (recommended 3.10–3.13)

PyTorch (with MPS/CUDA/CPU support)

Internet connection for model download (only on first launch)

TECH STACK

Python 3

PyTorch + OpenAI CLIP (ViT-B/32 model)

Telegram Bot API (pyTelegramBotAPI)

Pillow for image processing

CREDITS

Developed by Tomiris Zhagypar
AI-powered environmental project encouraging sustainable waste sorting.
