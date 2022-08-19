import socket
from datetime import datetime
from typing import TYPE_CHECKING, List

import telegram
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

if TYPE_CHECKING:
    from ..xgb import XGBBaseModel

from ...data.settings import get_notifier_settings

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TelegramBotCallback(Callback):
    def __init__(self):
        settings = get_notifier_settings()
        self.bot = telegram.Bot(token=settings.BOT_TOKEN)
        self.chat_id = settings.CHAT_ID
        self.host_name = socket.gethostname()
        self.n_epoch = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.fit_start_time = datetime.now()
        self.model_name = pl_module._get_name()
        self.max_epochs = pl_module.hparams.max_epochs
        self.val_losses = []
        hparams = pl_module.hparams
        self._send_initial_message(hparams)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.fit_end_time = datetime.now()
        self._send_final_message()

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.n_epoch += 1
        self.epoch_start = datetime.now()

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.n_epoch == 0:  # This means we are in a sanity check phase
            return None
        self.val_losses.append(val_loss := pl_module.avg_val_loss)
        epoch_duration = datetime.now() - self.epoch_start
        contents = [
            f"Validation finished for epoch {self.n_epoch} out of {self.max_epochs}",
            "Epoch duration: %s" % str(epoch_duration),
            "Obtained validation loss: %.2f" % val_loss,
        ]
        if self.n_epoch > 1:
            if val_loss < self.best_result:
                improvement = self.best_result - val_loss
                contents.append(
                    "New best result, with an improvement of: %.2f" % improvement
                )
                self.best_result = val_loss
            else:
                contents.append(
                    "Validation loss did not improve (best result: %.2f)"
                    % self.best_result
                )
        else:
            self.best_result = val_loss
        self._send_message(contents)

    def _send_message(self, contents: List[str]) -> None:
        text = "\n".join(contents)
        self.bot.send_message(chat_id=self.chat_id, text=text)

    def _send_initial_message(self, hparams) -> None:
        contents = [
            "Your training of the model %s has started 🎬" % self.model_name,
            "Machine name: %s" % self.host_name,
            "Starting date: %s" % self.fit_start_time.strftime(DATE_FORMAT),
            "Maximum number of epochs: %s" % self.max_epochs,
            f"Parameters of the model: {hparams}",
        ]
        self._send_message(contents)

    def _send_final_message(self) -> None:
        contents = [
            "Your training of the model %s has finished 🎉" % self.model_name,
            "Machine name: %s" % self.host_name,
            "Elapsed time: %s" % str(self.fit_end_time - self.fit_start_time),
            "Best result: %.2f" % self.best_result,
        ]
        self._send_message(contents)


class XGBTelegramBotCallback:
    def __init__(self):
        settings = get_notifier_settings()
        self.bot = telegram.Bot(token=settings.BOT_TOKEN)
        self.chat_id = settings.CHAT_ID
        self.host_name = socket.gethostname()
        self.n_model = 0

    def on_dataset_preparation(self, xgb_module: "XGBBaseModel", split: str) -> None:
        self.model_name = xgb_module._get_name()
        contents = [
            "Start time: %s" % datetime.now().strftime(DATE_FORMAT),
            f"Preparing the {split} dataset for the model {self.model_name}...",
        ]
        self._send_message(contents)

    def on_fit_start(self, xgb_module: "XGBBaseModel") -> None:
        self.fit_start_time = datetime.now()
        self.model_name = xgb_module._get_name()
        self.max_models = xgb_module.n_models
        self.val_losses = []
        hparams = xgb_module.hparams
        self._send_initial_message(hparams)

    def on_fit_end(self, xgb_module: "XGBBaseModel") -> None:
        self.fit_end_time = datetime.now()
        self.best_params = xgb_module.best_params
        self._send_final_message()

    def on_train_epoch_start(self) -> None:
        self.n_model += 1
        self.epoch_start = datetime.now()

    def on_validation_epoch_end(self, val_loss: float) -> None:
        if self.n_model == 0:  # This means we are in a sanity check phase
            return None
        self.val_losses.append(val_loss := val_loss)
        epoch_duration = datetime.now() - self.epoch_start
        contents = [
            f"Validation finished for model {self.n_model} out of {self.max_models}",
            "Epoch duration: %s" % str(epoch_duration),
            "Obtained validation loss: %.2f" % val_loss,
        ]
        if self.n_model > 1:
            if val_loss < self.best_result:
                improvement = self.best_result - val_loss
                contents.append(
                    "New best result, with an improvement of: %.2f" % improvement
                )
                self.best_result = val_loss
            else:
                contents.append(
                    "Validation loss did not improve (best result: %.2f)"
                    % self.best_result
                )
        else:
            self.best_result = val_loss
        self._send_message(contents)

    def _send_message(self, contents: List[str]) -> None:
        text = "\n".join(contents)
        self.bot.send_message(chat_id=self.chat_id, text=text)

    def _send_initial_message(self, hparams) -> None:
        contents = [
            "Your training of the model %s has started 🎬" % self.model_name,
            "Machine name: %s" % self.host_name,
            "Starting date: %s" % self.fit_start_time.strftime(DATE_FORMAT),
            "Maximum number of models: %s" % self.max_models,
            f"Parameters of the model: {hparams}",
        ]
        self._send_message(contents)

    def _send_final_message(self) -> None:
        contents = [
            "Your training of the model %s has finished 🎉" % self.model_name,
            "Machine name: %s" % self.host_name,
            "Elapsed time: %s" % str(self.fit_end_time - self.fit_start_time),
            "Best result: %.2f" % self.best_result,
            f"Optimal parameters: {self.best_params}",
        ]
        self._send_message(contents)
