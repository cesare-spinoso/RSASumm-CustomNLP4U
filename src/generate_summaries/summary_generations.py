import os

import datasets
import evaluate
import hydra
import jsonlines
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed

from src import SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.utils.decorators import main_decorator


def get_data(cfg):
    ds_name = cfg["dataset"]["name"]
    split_name = cfg["dataset"]["split"]
    source_name = cfg["dataset"]["source_name"]
    reference_name = cfg["dataset"]["reference_name"]
    if ds_name in ["cnn_dailymail"]:
        dataset = datasets.load_dataset(
            ds_name, cfg["dataset"]["version"], cache_dir=SCRATCH_CACHE_DIR
        )
        if split_name in ["train", "validation", "test"]:
            source = dataset[split_name][source_name]
            reference = dataset[split_name][reference_name]
        else:
            raise ValueError(f"Invalid or not implemented split {split_name} (e.g., all)")
    elif ds_name in ["covidet", "debatepedia", "multioped", "qmsum"]:
        ds_path = cfg["dataset"]["path"]
        dataset = pd.read_csv(ds_path)
        source = dataset[source_name]
        reference = dataset[reference_name]
    else:
        raise ValueError(f"Invalid dataset {ds_name}")
    return source, reference


def load_model(cfg):
    if cfg["tokenizer"]["name"] == cfg["model"]["name"] and cfg["model"][
        "name"
    ].startswith("t5"):
        # See https://github.com/huggingface/transformers/pull/24565 for why legacy=False
        tokenizer = T5Tokenizer.from_pretrained(
            cfg["tokenizer"]["name"], cache_dir=SCRATCH_CACHE_DIR, legacy=False
        )
        model = T5ForConditionalGeneration.from_pretrained(
            cfg["model"]["name"], cache_dir=SCRATCH_CACHE_DIR
        )
    return tokenizer, model


@hydra.main(
    version_base=None, config_path=os.path.join(SRC_DIRECTORY, "generate_summaries", "conf")
)
@main_decorator
def main(run_name: str, cfg: DictConfig) -> None:
    # Set seed if specified
    if "seed" in cfg["generation"]:
        set_seed(cfg["generation"]["seed"])
    # Load source and reference
    source, reference = get_data(cfg)
    # Load model
    tokenizer, model = load_model(cfg)
    # Tokenize source
    tokenized_source = tokenizer(
        ["summarize: " + s for s in source],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    source_input_ids = tokenized_source.input_ids
    # Generate by batch
    batch_size = cfg["generation"]["batch_size"]
    num_batches = source_input_ids.shape[0] // batch_size
    rouge = evaluate.load("rouge")
    for i in tqdm(range(num_batches)):
        # Generate
        outputs = model.generate(
            source_input_ids[i * batch_size : (i + 1) * batch_size],
            **cfg["generation"]["generate_kwargs"],
        )
        # Decode and calculate scores
        sources = [
            s
            for s in source[i * batch_size : (i + 1) * batch_size]
            for _ in range(
                cfg["generation"]["generate_kwargs"]["num_return_sequences"]
            )
        ]
        references = [
            r
            for r in reference[i * batch_size : (i + 1) * batch_size]
            for _ in range(
                cfg["generation"]["generate_kwargs"]["num_return_sequences"]
            )
        ]
        predictions = tokenizer.batch_decode(
            outputs["sequences"], skip_special_tokens=True
        )
        prediction_scores = outputs["sequences_scores"]
        assert (
            len(sources)
            == len(references)
            == len(predictions)
            == len(prediction_scores)
        )
        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=False,
        )
        assert all(
            len(prediction_scores) == len(rouge_scores[score_type])
            for score_type in rouge_scores
        )
        dict_to_write = [
            {
                "source": sources[i],
                "ref": references[i],
                "pred": predictions[i],
                "pred_score": prediction_scores[i].item(),
                **{
                    score_type: rouge_scores[score_type][i]
                    for score_type in rouge_scores
                },
            }
            for i in range(len(sources))
        ]
        with jsonlines.open(
            os.path.join(cfg["output_directory"], f"{run_name}.jsonl"), "a"
        ) as writer:
            writer.write_all(dict_to_write)


if __name__ == "__main__":
    main()
