import os

import hydra
import jsonlines
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from hydra.core.hydra_config import HydraConfig

from src import SRC_DIRECTORY, SCRATCH_CACHE_DIR
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines


def sanity_check_config(cfg):
    hydra_job = HydraConfig.get().job
    hydra_runtime = HydraConfig.get().runtime
    assert cfg["dataset_name"] == hydra_job["config_name"]
    assert (
        cfg["rescoring"]["type"] in cfg["write_config_directory"]
        and cfg["summarizer_name"] in cfg["write_config_directory"]
    )
    assert any(
        f"{cfg['rescoring']['type']}" in dict_elt["path"]
        for dict_elt in hydra_runtime["config_sources"]
    )
    assert any(
        f"{cfg['summarizer_name']}" in dict_elt["path"]
        for dict_elt in hydra_runtime["config_sources"]
    )


def get_latent_column_name(dataset_name):
    if dataset_name == "covidet":
        return "emotion"
    elif dataset_name == "debatepedia":
        return "topic"
    elif dataset_name == "duc_single":
        return "topic"
    elif dataset_name == "multioped":
        return "answer"
    elif dataset_name == "qmsum":
        return "question"
    else:
        raise ValueError(f"Dataset name {dataset_name} not recognized")


def load_model(cfg):
    model_name = cfg["model"]["name"]
    tokenizer_name = cfg["tokenizer"]["name"]
    if model_name == tokenizer_name and "t5" in model_name:
        model = (
            T5ForConditionalGeneration.from_pretrained(
                cfg["model"]["name"], cache_dir=SCRATCH_CACHE_DIR
            )
            .to("cuda")
            .half()
        )
        tokenizer = T5Tokenizer.from_pretrained(
            cfg["tokenizer"]["name"], cache_dir=SCRATCH_CACHE_DIR, legacy=False
        )
    else:
        raise ValueError(
            f"Invalid model/tokenizer combination: {model_name}/{tokenizer_name}"
        )
    model.eval()
    return model, tokenizer


def get_summary_jsonl(cfg):
    summary_jsonl_data = []
    with open(cfg["rescoring"]["generated_summaries_yaml"], "r") as yaml_file:
        try:
            generated_summaries = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    summary_jsonl_path = generated_summaries[cfg["summarizer_name"]][
        cfg["dataset_name"]
    ]
    with jsonlines.open(summary_jsonl_path) as reader:
        for obj in reader:
            summary_jsonl_data.append(obj)
    return summary_jsonl_data


def get_latent_rec_values(summary_jsonl_data, cfg):
    # Read from jsonl and csv and check that values are consistent
    document_ids = [obj["document_id"] for obj in summary_jsonl_data]
    source = [obj["source"] for obj in summary_jsonl_data]
    pred = [obj["pred"] for obj in summary_jsonl_data]
    assert len(document_ids) == len(source) == len(pred) == len(summary_jsonl_data)
    assert cfg["dataset_name"] in cfg["rescoring"]["preprocessed_dataset_path"]
    df = pd.read_csv(cfg["rescoring"]["preprocessed_dataset_path"])
    set_doc_ids = set(document_ids)
    set_source = set(source)
    assert len(set_doc_ids) == len(set_source), print(set_doc_ids, set_source)
    assert set_doc_ids <= set(df["document_id"].values.tolist()) and set_source <= set(
        df["document"].values.tolist()
    )
    # Create the combination of docs, latents and preds
    latent_column_name = get_latent_column_name(cfg["dataset_name"])
    latent_dict = df.groupby("document_id")[latent_column_name].apply(list).to_dict()
    output = []
    for doc_id, s, p in zip(document_ids, source, pred):
        # NOTE: The latent might have duplicates, this is an evaluation-time decision
        for latent in set(latent_dict[doc_id]):
            output.append((doc_id, s, p, latent))
    assert all(len(o) == len(output[0]) for o in output)
    return ([o[i] for o in output] for i in range(len(output[0])))


def get_source_rec_values(summary_jsonl_data, cfg):
    # input is what you will condition the reconstruction on
    # target is the target reconstruction which for the source
    # reconstruction is the source document
    rec_input = [obj["pred"] for obj in summary_jsonl_data]
    if cfg["summarizer_name"] != "oracle":
        # The target is what you'd like to reconstruct
        # The source is the LM's conditional
        doc_ids = [obj["document_id"] for obj in summary_jsonl_data]
        rec_target = [obj["source"] for obj in summary_jsonl_data]
        assert (
            len(rec_target) == len(rec_input) == len(doc_ids) == len(summary_jsonl_data)
        )
        # Sanity check the doc ids
        assert cfg["dataset_name"] in cfg["rescoring"]["preprocessed_dataset_path"]
        df = pd.read_csv(cfg["rescoring"]["preprocessed_dataset_path"])
        document_ids = set(df["document_id"].values.tolist())
        documents = set(df["document"].values.tolist())
        set_doc_ids = set(doc_ids)
        set_rec_target = set(rec_target)
        assert len(set_doc_ids) == len(set_rec_target), print(
            set_doc_ids, set_rec_target
        )
        assert set_doc_ids <= document_ids and set_rec_target <= documents, print(
            set_doc_ids, set_rec_target, document_ids, documents
        )
    elif cfg["summarizer_name"] == "oracle":
        raise NotImplementedError("Oracle implementation not done yet")
        assert "Will be different than the one above"
    else:
        raise ValueError("Summarizer name not recognized")
    return doc_ids, rec_input, rec_target


def get_output_size(tokenizer):
    if (
        hasattr(tokenizer, "name_or_path")
        and tokenizer.name_or_path == "google/flan-t5-large"
    ):
        return 32128
    else:
        raise ValueError("Tokenizer not recognized")


def compute_conditional_likelihoods(
    unnormalized_logits, tokenized_outputs, tokenizer, average=False
):
    assert unnormalized_logits.shape[1] == tokenized_outputs.shape[1]
    assert unnormalized_logits.shape[2] == get_output_size(tokenizer)
    # Normalize every set of logits
    softmax_logits = torch.log_softmax(unnormalized_logits, dim=-1)
    # Gather the logits for the target tokens : this creates a tensor of shape
    # batch size x seq_len x seq_len => take the diagonal to get the logits
    gathered_logits = torch.gather(
        input=softmax_logits,
        dim=2,
        index=tokenized_outputs[:, None, :]
        .to(int)
        .repeat(1, tokenized_outputs.shape[1], 1),
    )
    # dim1, dim2 as recommended by
    # https://pytorch.org/docs/1.9.0/generated/torch.diagonal.html#torch.diagonal
    diagonal_logits = torch.diagonal(gathered_logits, dim1=-2, dim2=-1)
    assert diagonal_logits.shape == tokenized_outputs.shape
    # Zero out the padding tokens
    zero_padding = torch.where(
        tokenized_outputs != tokenizer.pad_token_id, diagonal_logits, 0
    )
    conditional_likelihoods = zero_padding.sum(dim=-1)
    if average:
        conditional_likelihoods = conditional_likelihoods / (
            tokenized_outputs != tokenizer.pad_token_id
        ).sum(axis=-1)
    conditional_likelihoods = conditional_likelihoods.tolist()
    assert len(conditional_likelihoods) == unnormalized_logits.shape[0]
    return conditional_likelihoods


def get_likelihood_scores(
    model, tokenizer, batch_size, targets_tokenized, input_tokenized
):
    with torch.no_grad():
        unnormalized_logits = model(
            input_ids=input_tokenized, labels=targets_tokenized
        ).logits.to("cpu")
        # expect (batch_size, seq_len, vocab_size)
    assert (
        len(unnormalized_logits.shape) == 3
        and unnormalized_logits.shape[0] == batch_size
    )
    conditional_likelihoods = compute_conditional_likelihoods(
        unnormalized_logits,
        targets_tokenized.to("cpu"),
        tokenizer,
        average=False,
    )
    avg_conditional_likelihoods = compute_conditional_likelihoods(
        unnormalized_logits,
        targets_tokenized.to("cpu"),
        tokenizer,
        average=True,
    )

    return conditional_likelihoods, avg_conditional_likelihoods


def compute_source_reconstruction(run_name, cfg):
    # Load model and tokenizer and set to inference mode
    model, tokenizer = load_model(cfg)
    # Load data
    summary_jsonl_data = get_summary_jsonl(cfg)
    # Get reconstruction input and output
    doc_ids, rec_input, rec_target = get_source_rec_values(summary_jsonl_data, cfg)
    # Compute reconstruction
    # Run in batches
    batch_size = cfg["rescoring"]["batch_size"]
    num_batches = len(rec_target) // batch_size
    for i in tqdm(range(num_batches)):
        s = slice(i * batch_size, (i + 1) * batch_size)
        batch_doc_ids = doc_ids[s]
        batch_input = rec_input[s]
        batch_targets = rec_target[s]
        # NOTE: I verified this (for T5) no error is raised
        # if the input and target combined exceed the max length which
        # I think makes sense when you think of the transformer encoder/
        # decoder architecture
        targets_tokenized = tokenizer(
            batch_targets, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        input_tokenized = tokenizer(
            batch_input, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        # Compute the conditional log-likelihood for P(source|summary)
        # (value will be NEGATIVE)
        # NOTE: For T5, I have modified the transformers source code
        # to return a loss vector rather than a mean
        # This is to allow batching of the loss computation
        # I use a different method which does not use this hack
        conditional_likelihoods, avg_conditional_likelihoods = get_likelihood_scores(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            targets_tokenized=targets_tokenized,
            input_tokenized=input_tokenized,
        )
        # Write to new jsonl
        dict_to_write = [
            {
                "document_id": batch_doc_ids[i],
                "source": batch_targets[i],
                "pred": batch_input[i],
                "reconstruction_score": conditional_likelihoods[i],
                "avg_reconstruction_score": avg_conditional_likelihoods[i],
            }
            for i in range(batch_size)
        ]
        output_directory = os.path.join(
            cfg["output_directory"], cfg["rescoring"]["type"], cfg["summarizer_name"]
        )
        append_jsonlines(dict_to_write, output_directory, run_name)


def compute_latent_reconstruction(run_name, cfg):
    # Load model and tokenizer and set to inference mode
    model, tokenizer = load_model(cfg)
    # Load data
    summary_jsonl_data = get_summary_jsonl(cfg)
    # Get reconstruction input and output
    (doc_ids, sources, predictions, latents) = get_latent_rec_values(
        summary_jsonl_data, cfg
    )
    # Run in batches
    batch_size = cfg["rescoring"]["batch_size"]
    num_batches = len(doc_ids) // batch_size
    for i in tqdm(range(num_batches)):
        s = slice(i * batch_size, (i + 1) * batch_size)
        batch_doc_ids = doc_ids[s]
        batch_source = sources[s]
        batch_preds = predictions[s]
        batch_latents = latents[s]
        targets_tokenized = tokenizer(
            batch_latents, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        input_tokenized = tokenizer(
            batch_preds, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to("cuda")
        conditional_likelihoods, avg_conditional_likelihoods = get_likelihood_scores(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            targets_tokenized=targets_tokenized,
            input_tokenized=input_tokenized,
        )
        # Write to new jsonl
        dict_to_write = [
            {
                "document_id": batch_doc_ids[i],
                "source": batch_source[i],
                "pred": batch_preds[i],
                get_latent_column_name(cfg["dataset_name"]): batch_latents[i],
                "reconstruction_score": conditional_likelihoods[i],
                "avg_reconstruction_score": avg_conditional_likelihoods[i],
            }
            for i in range(batch_size)
        ]
        output_directory = os.path.join(
            cfg["output_directory"], cfg["rescoring"]["type"], cfg["summarizer_name"]
        )
        append_jsonlines(dict_to_write, output_directory, run_name)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        SRC_DIRECTORY, "rescoring", "conf"
    ),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    if cfg["rescoring"]["type"] == "source_reconstruction":
        compute_source_reconstruction(run_name, cfg)
    elif cfg["rescoring"]["type"] == "latent_reconstruction":
        compute_latent_reconstruction(run_name, cfg)
    elif cfg["rescoring"]["type"] == "qa_rescoring":
        raise NotImplementedError("QA rescoring not implemented")
    else:
        raise ValueError("Rescoring type not recognized")


if __name__ == "__main__":
    main()
