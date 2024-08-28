import os
import warnings

import hydra
import pandas as pd
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from src import HF_TOKEN, SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines, get_values_from_jsonlines, read_jsonlines
from src.utils.inference import batch_data


def sanity_check_config(cfg):
    hydra_job = HydraConfig.get().job
    hydra_runtime = HydraConfig.get().runtime
    assert cfg["dataset_name"] == hydra_job["config_name"]
    assert cfg["rescoring"]["type"] in cfg["write_config_directory"] and (
        cfg["summarizer_name"] in cfg["write_config_directory"]
        or "llama3" in cfg["write_config_directory"]
    )
    assert any(
        f"{cfg['rescoring']['type']}" in dict_elt["path"]
        for dict_elt in hydra_runtime["config_sources"]
    )
    assert any(
        f"{cfg['summarizer_name']}" in dict_elt["path"]
        or "llama3" in dict_elt["path"]
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
    elif dataset_name == "squality":
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
    elif model_name == tokenizer_name and "Llama-3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_size="left",
            token=HF_TOKEN,
            cache_dir=SCRATCH_CACHE_DIR,
        )
        tokenizer.pad_token = tokenizer.eos_token
        # Should use left padding since want the generation to immediately
        # start after the prompt
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            token=HF_TOKEN,
            cache_dir=SCRATCH_CACHE_DIR,
        )
    else:
        raise ValueError(
            f"Invalid model/tokenizer combination: {model_name}/{tokenizer_name}"
        )
    model.eval()
    return model, tokenizer


def get_summary_jsonl(cfg, run_name):
    summary_jsonl_data = []
    with open(cfg["rescoring"]["generated_summaries_yaml"], "r") as yaml_file:
        try:
            generated_summaries = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    summary_jsonl_path = generated_summaries[cfg["summarizer_name"]][
        cfg["dataset_name"]
    ]
    if isinstance(summary_jsonl_path, str):
        summary_jsonl_data = read_jsonlines(summary_jsonl_path)
    else:
        for jsonl_path in summary_jsonl_path:
            summary_jsonl_data += read_jsonlines(jsonl_path)
    if "rescored_summaries_jsonl_path" in cfg["rescoring"]:
        rescored_summaries = read_jsonlines(
            cfg["rescoring"]["rescored_summaries_jsonl_path"]
        )
        output_directory = os.path.join(
            cfg["output_directory"], cfg["rescoring"]["type"], cfg["summarizer_name"]
        )
        append_jsonlines(rescored_summaries, output_directory, run_name)
        if cfg["rescoring"]["type"] == "source_reconstruction":
            generated_source = [
                elt["source"] for elt in rescored_summaries
            ]
            summary_jsonl_data = [
                elt
                for elt in summary_jsonl_data
                if elt["source"] not in generated_source
            ]
            return summary_jsonl_data, None
        elif cfg["rescoring"]["type"] == "latent_reconstruction":
            return summary_jsonl_data, rescored_summaries
        elif cfg["rescoring"]["type"] == "answer_reconstruction":
            return summary_jsonl_data, rescored_summaries
        else:
            raise ValueError("Rescoring type not recognized")
    else:
        return summary_jsonl_data, None


def process_llama2_preds(rec_input):
    for i, elt in enumerate(rec_input):
        if "[/INST]" in elt:
            rec_input[i] = elt.split("[/INST]")[1].strip()
        elif "summary:" in elt:
            rec_input[i] = elt.split("summary:")[1].strip()
        else:
            rec_input[i] = elt
    return rec_input


def process_llama3_preds(preds: list[str], pred_type: str = "summary") -> list[str]:
    if pred_type == "summary":
        for i, elt in enumerate(preds):
            preds[i] = elt.split("Summary:assistant")[1].strip()
    elif pred_type == "qa":
        for i, elt in enumerate(preds):
            preds[i] = elt.split("Answer: assistant")[1].strip()
    return preds


def generic_latent_rec_values(summary_jsonl_data, rescored_summaries, cfg):
    # Get raw data for latent mapping
    df = pd.read_csv(cfg["rescoring"]["preprocessed_dataset_path"])
    latent_column_name = get_latent_column_name(cfg["dataset_name"])
    latent_dict = df.groupby("document")[latent_column_name].apply(list).to_dict()
    # Generate new data
    existing_pred_latent = []
    if rescored_summaries is not None:
        existing_pred_latent = [
            (elt["pred"], elt[latent_column_name]) for elt in rescored_summaries
        ]
    document_ids = []
    source = []
    pred = []
    latents = []
    for elt in summary_jsonl_data:
        # NOTE: The latent might have duplicates, this is an evaluation-time decision
        if elt["source"] not in latent_dict:
            continue
        for latent in set(latent_dict[elt["source"]]):
            if (elt["pred"], latent) in existing_pred_latent:
                continue
            document_ids.append(elt["document_id"])
            source.append(elt["source"])
            pred.append(elt["pred"])
            latents.append(latent)
    if "llama" in cfg["summarizer_name"]:
        pred = process_llama2_preds(pred)
    return document_ids, source, pred, latents


def e2e_latent_rec_values(summary_jsonl_data, rescored_summaries, cfg):
    # This assumes there is a one-to-one correspondence between
    # the question given to the e2e model and the latent for each dataset
    latent_column_name = get_latent_column_name(cfg["dataset_name"])
    # Use the raw dataframe to get the latent
    df_raw = pd.read_csv(cfg["rescoring"]["preprocessed_dataset_path"])
    if "covidet" == cfg["dataset_name"]:
        df_raw["question"] = df_raw["emotion"].apply(
            lambda x: cfg["question_template"].format(emotion=x)
        )
    if "debatepedia" == cfg["dataset_name"]:
        df_raw["question"] = df_raw["query"]
    question_to_latent = (
        df_raw.groupby("question")[latent_column_name].apply(list).to_dict()
    )
    # Generate new data
    existing_pred_latent = []
    if rescored_summaries is not None:
        existing_pred_latent = [
            (elt["pred"], elt[latent_column_name]) for elt in rescored_summaries
        ]
    document_ids = []
    source = []
    pred = []
    latents = []
    for elt in summary_jsonl_data:
        if elt["question"] not in question_to_latent:
            continue
        for latent in set(question_to_latent[elt["question"]]):
            if (elt["pred"], latent) in existing_pred_latent:
                continue
            document_ids.append(elt["document_id"])
            source.append(elt["source"])
            pred.append(elt["pred"])
            latents.append(latent)
    if "llama" in cfg["summarizer_name"]:
        pred = process_llama2_preds(pred)
    return document_ids, source, pred, latents


def get_latent_rec_values(summary_jsonl_data, rescored_summaries, cfg):
    if "generic" in cfg["summarizer_name"]:
        return generic_latent_rec_values(summary_jsonl_data, rescored_summaries, cfg)
    elif "e2e" in cfg["summarizer_name"]:
        return e2e_latent_rec_values(summary_jsonl_data, rescored_summaries, cfg)
    else:
        raise ValueError("Summarizer name not recognized")


def get_source_rec_values(summary_jsonl_data, cfg):
    # input is what you will condition the reconstruction on
    # target is the target reconstruction which for the source
    # reconstruction is the source document
    # The target is what you'd like to reconstruct
    # The source is the LM's conditional
    doc_ids = [obj["document_id"] for obj in summary_jsonl_data]
    rec_input = [obj["pred"] for obj in summary_jsonl_data]
    rec_target = [obj["source"] for obj in summary_jsonl_data]
    if "llama2" in cfg["summarizer_name"]:
        rec_input = process_llama2_preds(rec_input)
    elif "llama3" in cfg["summarizer_name"]:
        rec_input = process_llama3_preds(rec_input, pred_type="summary")
    if "prompt" in cfg["rescoring"]:
        rec_input = [
            cfg["rescoring"]["prompt"].format(s=summary) for summary in rec_input
        ]
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
    model, tokenizer, is_last_batch, batch_size, targets_tokenized, input_tokenized
):
    with torch.no_grad():
        unnormalized_logits = model(
            input_ids=input_tokenized, labels=targets_tokenized
        ).logits.to("cpu")
        # expect (batch_size, seq_len, vocab_size)
    assert len(unnormalized_logits.shape) == 3 and (
        is_last_batch or unnormalized_logits.shape[0] == batch_size
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
    summary_jsonl_data, _ = get_summary_jsonl(cfg, run_name)
    # Get reconstruction input and output
    doc_ids, rec_input, rec_target = get_source_rec_values(summary_jsonl_data, cfg)
    # Compute reconstruction
    # Run in batches
    batch_size = cfg["rescoring"]["batch_size"]
    num_batches = len(rec_target) // batch_size + (
        0 if len(rec_target) % batch_size == 0 else 1
    )
    for i in tqdm(range(num_batches)):
        s = slice(i * batch_size, (i + 1) * batch_size)
        batch_doc_ids = doc_ids[s]
        batch_input = rec_input[s]
        batch_targets = rec_target[s]
        # NOTE: I verified this (for T5) no error is raised
        # if the input and target combined exceed the max length which
        # I think makes sense when you think of the transformer encoder/
        # decoder architecture
        target_input_ids = tokenizer(
            batch_targets, padding=False, truncation=False, add_special_tokens=False
        )["input_ids"]
        target_shape = [len(x) for x in target_input_ids]
        input_shape = [
            len(x)
            for x in tokenizer(batch_input, padding=False, truncation=False)[
                "input_ids"
            ]
        ]
        concatenated_input_target = [
            input_ + target for input_, target in zip(batch_input, batch_targets)
        ]
        concatenated_input_ids = tokenizer(
            concatenated_input_target,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )["input_ids"].to("cuda")
        conditional_likelihoods, avg_conditional_likelihoods = get_qa_likelihood_scores(
            model=model,
            concatenated_input_ids=concatenated_input_ids,
            input_shape=input_shape,
            answer_input_ids=target_input_ids,
            answer_shape=target_shape,
        )
        # Write to new jsonl
        dict_to_write = [
            {
                "document_id": batch_doc_ids[i],
                "source": batch_targets[i],
                "pred": batch_input[i],
                "reconstruction_score": float(conditional_likelihoods[i]),
                "avg_reconstruction_score": float(avg_conditional_likelihoods[i]),
            }
            for i in range(len(batch_doc_ids))
        ]
        output_directory = os.path.join(
            cfg["output_directory"], cfg["rescoring"]["type"], cfg["summarizer_name"]
        )
        append_jsonlines(dict_to_write, output_directory, run_name)



def compute_latent_reconstruction(run_name, cfg):
    # Load model and tokenizer and set to inference mode
    model, tokenizer = load_model(cfg)
    # Load data
    summary_jsonl_data, rescored_summaries = get_summary_jsonl(cfg, run_name)
    # Get reconstruction input and output
    (doc_ids, sources, predictions, latents) = get_latent_rec_values(
        summary_jsonl_data, rescored_summaries, cfg
    )
    # Run in batches
    batch_size = cfg["rescoring"]["batch_size"]
    num_batches = len(doc_ids) // batch_size + (
        0 if len(doc_ids) % batch_size == 0 else 1
    )
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
            is_last_batch=(num_batches - 1 == i),
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
            for i in range(len(batch_doc_ids))
        ]
        output_directory = os.path.join(
            cfg["output_directory"], cfg["rescoring"]["type"], cfg["summarizer_name"]
        )
        append_jsonlines(dict_to_write, output_directory, run_name)


def get_answer_reconstruction_data(cfg, run_name):
    summary_jsonl_data = []
    with open(cfg["rescoring"]["generated_summaries_yaml"], "r") as yaml_file:
        try:
            generated_summaries = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    if "subkey" in cfg:
        summary_jsonl_path = generated_summaries[cfg["summarizer_name"]][
            cfg["dataset_name"]
        ][cfg["subkey"]]
    else:
        summary_jsonl_path = generated_summaries[cfg["summarizer_name"]][
            cfg["dataset_name"]
        ]
    summary_jsonl_data = read_jsonlines(summary_jsonl_path)
    if "rescored_summaries_jsonl_path" in cfg["rescoring"]:
        rescored_summaries = read_jsonlines(
            cfg["rescoring"]["rescored_summaries_jsonl_path"]
        )
        output_directory = os.path.join(
            cfg["output_directory"], cfg["rescoring"]["type"], cfg["summarizer_name"]
        )
        append_jsonlines(rescored_summaries, output_directory, run_name)
        generated_source_pred = [
            (elt["summary"], elt["source"], elt["question"]) for elt in rescored_summaries
        ]
        summary_jsonl_data = [
            elt
            for elt in summary_jsonl_data
            if (elt["pred"], elt["source"], elt["question"]) not in generated_source_pred
        ]
    document_ids, documents, summaries, questions = get_values_from_jsonlines(
        jsonlines_data=summary_jsonl_data,
        keys=["document_id", "source", "pred", "question"],
    )
    if cfg["summarizer_name"].startswith("llama3"):
        summaries = process_llama3_preds(summaries, pred_type="summary")
    answers_jsonl_path = cfg["rescoring"]["generated_answers_path"]
    answers_data = read_jsonlines(answers_jsonl_path)
    answers_doc_id, answers_questions, generated_answers = get_values_from_jsonlines(
        jsonlines_data=answers_data,
        keys=["document_id", "question", "generated_answer"],
    )
    if (qa_name := cfg.get("qa_name")) == "llama3_qa":
        generated_answers = process_llama3_preds(generated_answers, pred_type="qa")
    else:
        raise ValueError(f"QA name {qa_name} not recognized")
    answers_dict = dict(
        zip(
            [
                (ans_id, ans_q)
                for ans_id, ans_q in zip(answers_doc_id, answers_questions)
            ],
            generated_answers,
        )
    )

    return (
        document_ids,
        documents,
        summaries,
        questions,
        [answers_dict[(doc_id, q)] for doc_id, q in zip(document_ids, questions)],
    )


def get_qa_likelihood_scores(
    model, concatenated_input_ids, input_shape, answer_input_ids, answer_shape
):
    output_logits = model(concatenated_input_ids).logits.to("cuda")
    sliced_normalized_logits = []
    for tensor_2d, start, offset, answer_ids in zip(
        output_logits, input_shape, answer_shape, answer_input_ids
    ):
        sliced_normalized_logits += [
            torch.gather(
                input=torch.log_softmax(tensor_2d[start : (start + offset), :], dim=1),
                dim=1,
                index=torch.tensor(answer_ids)[:, None].to("cuda"),
            ).sum()
        ]
    sliced_normalized_logits = torch.tensor(sliced_normalized_logits)
    answer_shape = torch.tensor(answer_shape)
    return sliced_normalized_logits.to("cpu"), (
        sliced_normalized_logits / answer_shape
    ).to("cpu")


def combine_inputs_for_qa(
    prompt: str, summaries: list[str], questions: list[str]
) -> list[str]:
    combined = []
    for s, q in zip(summaries, questions):
        combined.append(prompt.format(d=s, q=q))
    return combined


def compute_qa_reconstruction(run_name, cfg):
    # Load model and tokenizer and set to inference mode
    model, tokenizer = load_model(cfg)
    # Load data
    document_ids, documents, summaries, questions, answers = (
        get_answer_reconstruction_data(cfg, run_name)
    )
    inputs = combine_inputs_for_qa(
        prompt=cfg["rescoring"]["prompt"],
        summaries=summaries,
        questions=questions,
    )
    # Run in batches
    batch_size = cfg["rescoring"]["batch_size"]
    num_batches = len(document_ids) // batch_size + (
        0 if len(document_ids) % batch_size == 0 else 1
    )
    for i in tqdm(range(num_batches)):
        (
            batch_doc_ids,
            batch_docs,
            batch_summaries,
            batch_questions,
            batch_answer,
            batch_inputs,
        ) = batch_data(
            data=[document_ids, documents, summaries, questions, answers, inputs],
            batch_size=batch_size,
            idx=i,
        )
        answer_input_ids = tokenizer(
            batch_answer, padding=False, truncation=False, add_special_tokens=False
        )["input_ids"]
        answer_shape = [len(x) for x in answer_input_ids]
        input_shape = [
            len(x)
            for x in tokenizer(batch_inputs, padding=False, truncation=False)[
                "input_ids"
            ]
        ]
        concatenated_input_answer = [
            input_ + answer for input_, answer in zip(batch_inputs, batch_answer)
        ]
        concatenated_input_ids = tokenizer(
            concatenated_input_answer,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )["input_ids"].to("cuda")
        conditional_likelihoods, avg_conditional_likelihoods = get_qa_likelihood_scores(
            model=model,
            concatenated_input_ids=concatenated_input_ids,
            input_shape=input_shape,
            answer_input_ids=answer_input_ids,
            answer_shape=answer_shape,
        )
        # Write to new jsonl
        dict_to_write = [
            {
                "document_id": batch_doc_ids[i],
                "source": batch_docs[i],
                "question": batch_questions[i],
                "summary": batch_summaries[i],
                "answer": batch_answer[i],
                "reconstruction_score": float(conditional_likelihoods[i]),
                "avg_reconstruction_score": float(avg_conditional_likelihoods[i]),
            }
            for i in range(len(batch_doc_ids))
        ]
        output_directory = os.path.join(
            cfg["output_directory"], cfg["rescoring"]["type"], cfg["summarizer_name"]
        )
        append_jsonlines(dict_to_write, output_directory, run_name)


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        SRC_DIRECTORY,
        "rescoring",
        "conf",
    ),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    if cfg["rescoring"]["type"] == "source_reconstruction":
        compute_source_reconstruction(run_name, cfg)
    # TODO: Add option if prediction was also conditioned on latent
    elif cfg["rescoring"]["type"] == "latent_reconstruction":
        compute_latent_reconstruction(run_name, cfg)
    elif cfg["rescoring"]["type"] == "answer_reconstruction":
        compute_qa_reconstruction(run_name, cfg)
    else:
        raise ValueError("Rescoring type not recognized")


if __name__ == "__main__":
    main()
