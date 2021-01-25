import logging
import os

from decouple import config
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler

from utils import AudioWorker, FeatureExtractor, Predictor

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

logger = logging.getLogger(__name__)
predictor = Predictor()
fx = FeatureExtractor()


def download_file(file):
    file_path = file.file_path.rsplit("/", 1)[-1]
    file.download()
    return file_path


def start(update: Update, context: CallbackContext):
    text = """I'm a Emotier - bot, which can recognize your emotion by voice message.\n\n Please, send me a one!"""
    context.bot.send_message(chat_id=update.effective_chat.id, text=text,)


def echo(update: Update, context: CallbackContext):
    user = update.message.from_user
    logger.info(f"User {user.first_name} sent a message {update.message.text}")
    text = """Unfortunately, i can't recognize emotions via text message yet.\nPlease, send me a voice message or audiofile with your voice."""
    context.bot.send_message(chat_id=update.effective_chat.id, text=text)


def voice_reply(update: Update, context: CallbackContext):
    user = update.message.from_user
    logger.info(f"User {user.first_name} send a voice/sound.")
    context.bot.send_message(chat_id=update.effective_chat.id, text="I'd receive your voice, thanks! Start processing...")
    file_path = download_file(context.bot.get_file(update.message.voice.file_id))
    feature_vector = AudioWorker().fit(file_path)
    feature_vector = fx.fit_transform(feature_vector)
    out = predictor.predict(feature_vector)
    logger.info(f"Out for user {user.first_name} is {out}")

    output="Oh! I know... Your emotion is (top 3):\n"
    for k, v in sorted(out.items(), key=lambda x: x[1], reverse=True)[:3]:
        output += f"- in {100*v:.2f}% is {k.upper()}\n"
    output += "Am i right? :)"
    context.bot.send_message(chat_id=update.effective_chat.id, text=output)


def cancel(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Bye! I hope we can talk again some day.'
    )

    return ConversationHandler.END


def main() -> None:
    TOKEN = config("TOKEN")
    NAME = config("NAME")

    PORT = os.environ.get("PORT")

    updater = Updater(token=TOKEN)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler("start", start)
    echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)
    voice_handler = MessageHandler(Filters.voice | Filters.audio, voice_reply)
    cancel_handler = CommandHandler("end", cancel)

    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(echo_handler)
    dispatcher.add_handler(voice_handler)
    dispatcher.add_handler(cancel_handler)

    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)
    updater.bot.setWebhook(f"https://{NAME}.herokuapp.com/{TOKEN}")
    updater.idle()


if __name__ == '__main__':
    main()