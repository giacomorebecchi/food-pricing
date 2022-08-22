import logging

from telegram.bot import Bot

from ...data.settings import get_notifier_settings


def send_telegram_message(message: str) -> None:
    try:
        bot_settings = get_notifier_settings()
        bot = Bot(token=bot_settings.BOT_TOKEN)
        bot.send_message(chat_id=bot_settings.CHAT_ID, text=message)
    except Exception as e:
        logging.error(
            f"Failed to send message: {message} with Telegram Bot.\nException: {e}"
        )
