import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from src import SRC_DIRECTORY
from src.utils.decorators import main_decorator
import jsonlines
from transformers import T5Tokenizer, T5ForConditionalGeneration


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rescoring", "conf")
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    # Load data
    summary_jsonl_data = []
    with jsonlines.open(cfg["rescoring"]["summary_jsonl_path"]) as reader:
        for obj in reader:
            summary_jsonl_data.append(obj)
    predictions = [obj["pred"] for obj in summary_jsonl_data]
    sources = [obj["source"] for obj in summary_jsonl_data]
    assert len(predictions) == len(sources) == len(summary_jsonl_data)
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(cfg["model"]["name"])
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(cfg["tokenizer"]["name"], legacy=False)
    # Run in batches
    batch_size = cfg["rescoring"]["batch_size"]
    num_batches = len(predictions) // batch_size
    with jsonlines.open(
        os.path.join(cfg["output_directory"], f"{run_name}.jsonl"), "a"
    ) as writer:
        for i in tqdm(range(num_batches)):
            batch_predictions = predictions[i * batch_size : (i + 1) * batch_size]
            batch_sources = sources[i * batch_size : (i + 1) * batch_size]
            predictions_tokenized = tokenizer(
                batch_predictions, padding=True, truncation=True, return_tensors="pt"
            ).input_ids
            sources_tokenized = tokenizer(
                batch_sources, padding=True, truncation=True, return_tensors="pt"
            ).input_ids
            # Compute the NLL for P(source|summary)
            # NOTE: I have modified the transformers source code to return a loss vector rather than a mean
            # This is to allow batching of the loss computation
            with torch.no_grad():
                model_losses = model(
                    input_ids=predictions_tokenized, labels=sources_tokenized
                ).loss
            avg_losses = model_losses.reshape(batch_size, -1).mean(dim=1).tolist()
            assert len(avg_losses) == batch_size
            # Write to new jsonl
            for j, loss in enumerate(avg_losses):
                summary_jsonl_data[i * batch_size + j]["loss_source_given_pred"] = loss
            writer.write_all(summary_jsonl_data[i * batch_size : (i + 1) * batch_size])


if __name__ == "__main__":
    main()
