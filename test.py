import logging

from decouple import config
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from utils import AudioWorker

TOKEN = config("TOKEN")

updater = Updater(token=TOKEN)
dispatcher = updater.dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

def download_file(file):
    file_path = file.file_path.rsplit("/", 1)[-1]
    file.download()
    return file_path


def start(update, context):
    text = """I'm a Emotier - bot, which can recognize your emotion by voice message. Please, send me a one!"""
    context.bot.send_message(chat_id=update.effective_chat.id, text=text)


def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


def voice_reply(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'd receive your voice, thanks! Start processing...")
    file_path = download_file(context.bot.get_file(update.message.voice.file_id))
    feature_vector = AudioWorker().fit(file_path)

    context.bot.send_message(chat_id=update.effective_chat.id, text="I'd receive your voice, thanks!")


start_handler = CommandHandler("start", start)
echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)
voice_handler = MessageHandler(Filters.voice | Filters.audio, voice_reply)

dispatcher.add_handler(start_handler)
dispatcher.add_handler(echo_handler)
dispatcher.add_handler(voice_handler)


if __name__ == '__main__':
    updater.start_polling()