import logging
import traceback

from src.models.base_model import FoodPricingBaseModel
from src.models.baseline import FPCBOWResNet152ConcatBaselineModel, FPMeanBaselineModel
from src.models.bert import (
    FPBERTResNet152ConcatModel,
    FPBERTResNet152WeightedConcatModel,
)
from src.models.clip import FPCLIPConcatModel, FPCLIPWeightedConcatModel
from src.models.utils.notifier import (
    ModelsList,
    get_init_message,
    send_telegram_message,
)
from src.models.utils.storage import get_hparams_config, get_log_path, get_run_id
from src.models.xgb import XGBCLIP, XGBBaseModel, XGBBERTResNet152

logging.basicConfig(
    filename=get_log_path(),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s",
)

MODELS = [
    FPMeanBaselineModel,
    FPCBOWResNet152ConcatBaselineModel,
    FPBERTResNet152ConcatModel,
    FPCLIPConcatModel,
    FPBERTResNet152WeightedConcatModel,
    FPCLIPWeightedConcatModel,
    XGBBERTResNet152,
    XGBCLIP,
]


def main(
    models: ModelsList = MODELS,
) -> None:

    run_id = get_run_id()
    hparams_config = get_hparams_config()
    init_message = get_init_message(models=models, hparams_config=hparams_config)
    send_telegram_message(init_message)

    for model_class in models:
        model_name = model_class.__name__
        if issubclass(model_class, FoodPricingBaseModel):
            hparams = hparams_config["pytorch"]
        elif issubclass(model_class, XGBBaseModel):
            hparams = hparams_config["xgb"]
        elif issubclass(model_class, FPMeanBaselineModel):
            hparams = {}
        else:
            raise Exception(f"Unrecognised class {model_name}.")

        hparams["trainer_run_id"] = run_id

        # Create the model instance
        logging.info(f"Creating instance of model {model_name}.")
        model = model_class(**hparams)

        # Fit the model
        logging.info(f"Starting to fit model {model_name}.")
        model.fit()
        logging.info(f"Finished fitting model {model_name}.")

        # Produce predictions (stored in ./submissions)
        _ = model.make_submission_frame()
        logging.info("Finished submitting predictions for model {model_name}.")

        # Free memory
        del model
        logging.info("Released memory occupied from model {model_name}.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        trbck = traceback.format_exc()
        message = (
            f"Training interrupted at time {get_run_id()}.\n"
            + f"Complete traceback: {trbck}"
        )
        send_telegram_message(message)
        logging.error(trbck)
