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
from src.models.utils.storage import get_hparams_config, get_run_id
from src.models.xgb import XGBCLIP, XGBBaseModel, XGBBERTResNet152

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
    init_message = get_init_message()
    send_telegram_message(init_message)

    for model_class in models:
        if issubclass(model_class, FoodPricingBaseModel):
            hparams = hparams_config["pytorch"]
        elif issubclass(model_class, XGBBaseModel):
            hparams = hparams_config["xgb"]
        elif issubclass(model_class, FPMeanBaselineModel):
            hparams = {}
        else:
            raise Exception(f"Unrecognised class {model_class}")

        hparams["trainer_run_id"] = run_id

        # Create the model instance
        model = model_class(**hparams)

        # Fit the model
        model.fit()

        # Produce predictions (stored in ./submissions)
        _ = model.make_submission_frame()

        # Free memory
        del model


if __name__ == "__main__":
    try:
        main()
    except Exception:
        message = f"Training interrupted at time {get_run_id()}.\n"
        f"Complete traceback: {traceback.format_exc()}"
        send_telegram_message(message)
        print(traceback.format_exc())
