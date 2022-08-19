import socket
from datetime import datetime
from typing import List

import telegram
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from ...data.settings import get_notifier_settings


class TelegramBotCallback(Callback):
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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
            "Starting date: %s" % self.fit_start_time.strftime(self.DATE_FORMAT),
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
