import os
import warnings

import hydra
import pytorch_lightning as pl
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping
from transformers import AutoTokenizer

from src import SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.finetuning.literal_summarizer.data import LiteralSummarizerDataModule
from src.finetuning.literal_summarizer.models import LiteralSummarizerPLModule
from src.utils.decorators import main_decorator


def sanity_check_config(cfg):
    hydra_job = HydraConfig.get().job
    hydra_runtime = HydraConfig.get().runtime
    assert (
        hydra_job["config_name"] in cfg["data"]["name"]
        and hydra_job["config_name"] in cfg["data"]["csv_dir"]
    )
    assert "bart" in cfg["model"]["name"] and "bart" in hydra_runtime["config_sources"]
    if "sample" in cfg["data"]["csv_dir"]:
        warnings.warn(
            "RUNNING ON SAMPLE DATA. The resulting checkpoints should not be used."
        )


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "finetuning", "literal_summarizer"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig) -> None:
    output_directory = os.path.join(cfg["output_directory"], run_name)
    os.makedirs(output_directory, exist_ok=True)
    # Wandb
    wandb.init(
        project="RSASumm",
        name=run_name,
        config=cfg,
        dir=output_directory,
    )
    # Load data
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"], cache_dir=SCRATCH_CACHE_DIR
    )
    data_module = LiteralSummarizerDataModule(
        dataset_name=cfg["data"]["name"],
        csv_dir=cfg["data"]["csv_dir"],
        tokenizer=tokenizer,
        dataset_formatting=cfg["data"]["dataset_formatting"],
        train_batch_size=cfg["hps"]["train_batch_size"],
        test_batch_size=cfg["hps"]["test_batch_size"],
    )
    data_module.prepare_data()
    data_module.setup("fit")
    # Train model
    t_total = (
        len(data_module.train_dataloader())
        // cfg["hps"]["gradient_accumulation_steps"]
        * cfg["hps"]["epochs"]
    )
    pl_module = LiteralSummarizerPLModule(
        model_name=cfg["model"]["name"],
        tokenizer=tokenizer,
        learning_rate=cfg["hps"]["learning_rate"],
        weight_decay=cfg["hps"]["weight_decay"],
        warmup=cfg["hps"]["warmup"],
        training_steps=t_total,
        is_logging_rouge_scores=cfg["model"]["is_logging_rouge_scores"],
        lora=cfg.get("lora", None),
        test_batch_size=cfg["hps"].get("test_batch_size", None),
        test_interval_computation=cfg["model"].get("test_interval_computation", None),
        test_num_batch_sample=cfg["model"].get("test_num_batch_sample", None),
        generation_kwargs=cfg["model"].get("generation_kwargs", None),
    )
    # TODO: Add checkpointing in case gets interrupted
    trainer = pl.Trainer(
        default_root_dir=output_directory,
        accelerator="gpu",
        devices=1,
        max_epochs=cfg["hps"]["epochs"],
        callbacks=[
            EarlyStopping(monitor="avg_val_loss", patience=cfg["hps"]["patience"])
        ],
        log_every_n_steps=1,
        accumulate_grad_batches=cfg["hps"][
            "gradient_accumulation_steps"
        ],  # Defaults to 1
    )
    trainer.fit(pl_module, datamodule=data_module)
    cfg["best_model_path"] = trainer.checkpoint_callback.best_model_path
    wandb.finish()


if __name__ == "__main__":
    """
    NOTE: To use the sweeping functionality add:
    hydra:
        sweeper:
            params:
                hps.lr: 1,2
    AND enable multirun with --multirun
    This should create multiple run instances (with different names created by the decorator).
    """
    main()
