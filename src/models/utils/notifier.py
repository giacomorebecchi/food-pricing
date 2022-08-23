import logging
from typing import List, Union

from telegram.bot import Bot

from ...data.settings import get_notifier_settings
from ..base_model import FoodPricingBaseModel
from ..baseline import FPMeanBaselineModel
from ..xgb import XGBBaseModel
from .storage import ParamConfig

ModelsList = List[Union[XGBBaseModel, FoodPricingBaseModel, FPMeanBaselineModel]]


def send_telegram_message(message: str) -> None:
    try:
        bot_settings = get_notifier_settings()
        bot = Bot(token=bot_settings.BOT_TOKEN)
        bot.send_message(chat_id=bot_settings.CHAT_ID, text=message)
    except Exception as e:
        logging.error(
            f"Failed to send message: {message} with Telegram Bot.\nException: {e}"
        )


def get_init_message(models: ModelsList, hparams_config: ParamConfig) -> str:
    init_message = "\n".join(
        ["Starting the training of models:"]
        + [m.__name__ for m in models]
        + [""]
        + ["The following hyperparameters have been selected:"]
        + ["Pytorch models:"]
        + [
            "\t" + str(key) + ": " + str(value)
            for key, value in hparams_config["pytorch"].items()
        ]
        + ["XGB models:"]
        + [
            "\t" + str(key) + ": " + str(value)
            for key, value in hparams_config["xgb"].items()
        ]
    )
    return init_message
