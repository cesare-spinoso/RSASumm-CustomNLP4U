from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import set_seed
import datasets
import evaluate
import jsonlines
import hydra
from tqdm import tqdm
import os
from src import SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.utils.decorators import main_decorator
from omegaconf import DictConfig

CNNDM = "cnn_dailymail"
MODEL_NAME = "google-t5/t5-small"


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "generate_data", "conf")
)
@main_decorator
def main(run_name: str, cfg: DictConfig) -> None:
    # Load dataset
    # Use 3.0.0 since it's the same as Andreas's paper
    dataset = datasets.load_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["version"], cache_dir=SCRATCH_CACHE_DIR
    )
    # Load model
    # See https://github.com/huggingface/transformers/pull/24565 for why legacy=False
    tokenizer = T5Tokenizer.from_pretrained(
        cfg["tokenizer"]["name"], cache_dir=SCRATCH_CACHE_DIR, legacy=False
    )
    model = T5ForConditionalGeneration.from_pretrained(
        cfg["model"]["name"], cache_dir=SCRATCH_CACHE_DIR
    )
    # Get test data
    test_source = dataset["test"][cfg["dataset"]["source_name"]]
    test_reference = dataset["test"][cfg["dataset"]["reference_name"]]
    tokenized_source = tokenizer(
        ["summarize: " + s for s in test_source],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    source_input_ids = tokenized_source.input_ids
    # Generate by batch
    batch_size = cfg["generation"]["batch_size"]
    num_batches = source_input_ids.shape[0] // batch_size
    rouge = evaluate.load("rouge")
    # Set seed if specified
    if "seed" in cfg["generation"]:
        set_seed(cfg["generation"]["seed"])
    with jsonlines.open(
        os.path.join(cfg["output_directory"], f"{run_name}.jsonl"), "a"
    ) as writer:
        for i in tqdm(range(num_batches)):
            # Generate
            outputs = model.generate(
                source_input_ids[i * batch_size : (i + 1) * batch_size],
                **cfg["generation"]["generate_kwargs"],
            )
            # Decode and calculate scores
            sources = [
                s
                for s in test_source[i * batch_size : (i + 1) * batch_size]
                for _ in range(
                    cfg["generation"]["generate_kwargs"]["num_return_sequences"]
                )
            ]
            references = [
                r
                for r in test_reference[i * batch_size : (i + 1) * batch_size]
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
            writer.write_all(dict_to_write)


if __name__ == "__main__":
    main()
