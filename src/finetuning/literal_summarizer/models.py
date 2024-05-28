import evaluate
import pytorch_lightning as pl
import torch
import wandb
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    LEDForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from src import SCRATCH_CACHE_DIR


class LiteralSummarizerPLModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        learning_rate: float,
        weight_decay: float,
        warmup: float,
        training_steps: int,
        is_logging_rouge_scores: bool,
        lora: dict = None,
        test_batch_size: int = None,
        test_interval_computation: int = None,
        test_num_batch_sample: int = None,
        generation_kwargs: dict = None,
    ):
        super().__init__()
        # NOTE: This is quite important for when you want to seamlessly
        # load the model from a checkpoint
        # See: https://github.com/Lightning-AI/pytorch-lightning/issues/3981
        self.save_hyperparameters()
        if "bart" in model_name:
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name, cache_dir=SCRATCH_CACHE_DIR
            ).to("cuda")
        elif "led" in model_name:
            self.model = LEDForConditionalGeneration.from_pretrained(
                model_name, cache_dir=SCRATCH_CACHE_DIR
            ).to("cuda")
        else:
            raise ValueError("Only BART models are supported, for now.")
        if lora is not None:
            # NOTE: Lora on smaller models does not work very well
            self._print_trainable_parameters()
            lora_config = LoraConfig(**lora)
            self.model = get_peft_model(self.model, lora_config)
            self._print_trainable_parameters()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if isinstance(warmup, float):
            self.warmup_steps = int(warmup * training_steps)
        else:
            self.warmup_steps = warmup
        self.training_steps = training_steps
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.is_logging_rouge_scores = is_logging_rouge_scores
        if self.is_logging_rouge_scores:
            self.test_batch_size = test_batch_size
            self.test_interval_computation = test_interval_computation
            self.test_num_batch_samples = test_num_batch_sample
            self.generation_kwargs = generation_kwargs
            self.rouge_evaluator = evaluate.load("rouge")
            # Log outputs as well if this is the case
            self.wandb_table = wandb.Table(
                columns=["source", "reference", "prediction"]
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        global_attention_mask=None,
    ):
        if global_attention_mask is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                global_attention_mask=global_attention_mask,
            )

    def _print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        https://huggingface.co/docs/peft/main/en/task_guides/semantic_segmentation_lora#load-a-base-model
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params}"
            + f"|| all params: {all_param} ||"
            + f"trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def _step(self, batch, fold):
        # Collect values from batch
        source_ids = batch["source_ids"]
        source_mask = batch["source_mask"]
        label_ids = batch["label_ids"]
        # Zero out the padding tokens:
        # https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb
        decoder_input_ids = label_ids.clone()
        decoder_input_ids[decoder_input_ids[:, :] == self.tokenizer.pad_token_id] = -100
        # If using LED, use the global attention mask as well
        # Set it to the first token as done in the notebook here
        # https://huggingface.co/allenai/led-large-16384 (also what the paper recommends for summarization)
        global_attention_mask = None
        if "led" in self.model_name:
            global_attention_mask = torch.zeros_like(source_ids)
            global_attention_mask[:, 0] = 1
        # Run forward pass
        outputs = self(
            input_ids=source_ids,
            attention_mask=source_mask,
            labels=decoder_input_ids,
            global_attention_mask=None,
        )
        loss = outputs.loss
        # Logging
        self.log(f"{fold}_loss", loss)
        wandb.log({f"{fold}_loss": loss})
        # For end of epoch computation
        to_append = {
            "loss": loss,
            "source_ids": source_ids,
            "source_mask": source_mask,
            "label_ids": label_ids,
        }
        if fold == "train":
            self.training_step_outputs.append(to_append)
        elif fold == "val":
            self.validation_step_outputs.append(to_append)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        return loss

    def _log_avg_loss(self, fold):
        if fold == "train":
            outputs = self.training_step_outputs
        elif fold == "val":
            outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(f"avg_{fold}_loss", avg_loss)
        wandb.log({f"avg_{fold}_loss": avg_loss})

    def generate_summaries(self, source_ids, source_mask):
        num_batches = source_ids.shape[0] // self.test_batch_size + (
            1 if source_ids.shape[0] % self.test_batch_size != 0 else 0
        )
        summaries = []
        with torch.no_grad():
            for i in range(num_batches):
                batch_source_ids = source_ids[
                    i * self.test_batch_size : (i + 1) * self.test_batch_size
                ]
                batch_attention_mask = source_mask[
                    i * self.test_batch_size : (i + 1) * self.test_batch_size
                ]
                predictions = self.model.generate(
                    batch_source_ids,
                    attention_mask=batch_attention_mask,
                    **self.generation_kwargs,
                )
                summaries.extend(
                    self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                )
        return summaries

    def _log_rouge_scores(self, fold):
        if (
            not self.is_logging_rouge_scores
            or self.current_epoch % self.test_interval_computation != 0
        ):
            return
        if fold == "train":
            outputs = self.training_step_outputs
        elif fold == "val":
            outputs = self.validation_step_outputs
        # Log rouge scores (on a sample from the validation set)
        num_test_samples = self.test_batch_size * self.test_num_batch_samples
        source_ids = torch.cat([x["source_ids"] for x in outputs], dim=0)[
            :num_test_samples
        ]
        source_mask = torch.cat([x["source_mask"] for x in outputs], dim=0)[
            :num_test_samples
        ]
        generated_summaries = self.generate_summaries(source_ids, source_mask)
        label_ids = torch.cat([x["label_ids"] for x in outputs], dim=0)[
            :num_test_samples
        ]
        reference_summaries = self.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        rouge_scores = self.rouge_evaluator.compute(
            predictions=generated_summaries,
            references=reference_summaries,
            use_aggregator=True,
        )
        wandb.log({f"{fold}_rouge_1": rouge_scores["rouge1"]})
        wandb.log({f"{fold}_rouge_2": rouge_scores["rouge2"]})
        wandb.log({f"{fold}_rouge_L": rouge_scores["rougeL"]})
        # Log (same) example prediction
        source_text = self.tokenizer.decode(source_ids[0, :], skip_special_tokens=True)
        self.wandb_table.add_data(
            source_text, reference_summaries[0], generated_summaries[0]
        )
        wandb.log({"example_table": self.wandb_table})

    def on_train_epoch_end(self):
        self._log_avg_loss(fold="train")
        # Free memory as: https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self._log_avg_loss(fold="val")
        self._log_rouge_scores(fold="val")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # No weight decay for layer norm or bias:
        # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L1000
        # Prepare optimizer and schedule (linear warmup and decay)
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )
        self.opt = optimizer
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.training_steps,
        )
        return [optimizer], [scheduler]
