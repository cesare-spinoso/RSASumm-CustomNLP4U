import os

import datasets
import hydra
import numpy as np
import pandas as pd
import stanza
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    LEDForConditionalGeneration,
    LEDTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)

from src import HF_TOKEN, SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.evaluation.compute_metrics import compute_rouge
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines, read_jsonlines


def get_data_e2e(cfg, run_name):
    ds_name = cfg["dataset"]["name"]
    source_name = cfg["dataset"]["source_name"]
    question_name = cfg["dataset"]["question_name"]
    ds_path = cfg["dataset"]["path"]
    dataset = pd.read_csv(ds_path)
    if ds_name == "covidet":
        # Create the question
        assert question_name == "question"
        question_template = cfg["dataset"]["question_template"]
        dataset["question"] = dataset["emotion"].apply(
            lambda emotion: question_template.format(emotion=emotion)
        )
    duplicated_documents = dataset[[source_name, question_name]].duplicated(
        keep="first"
    )
    dataset = dataset[~duplicated_documents]
    if "generated_jsonl_paths" in cfg["generation"]:
        generated_jsonlines = []
        for jsonlines_path in cfg["generation"]["generated_jsonl_paths"]:
            generated_jsonlines += read_jsonlines(jsonlines_path)
        # Write existing to the jsonl file
        write_to_jsonlines = []
        for elt in generated_jsonlines:
            if elt not in write_to_jsonlines:
                write_to_jsonlines.append(elt)
        append_jsonlines(
            write_to_jsonlines,
            output_directory=cfg["output_directory"],
            run_name=run_name,
        )
        # Filter for existing generation
        generated_questions = [elt["question"] for elt in generated_jsonlines]
        generated_sources = [elt["source"] for elt in generated_jsonlines]
        generated_combined = list(zip(generated_questions, generated_sources))
        already_generated = [
            (q, s) in generated_combined
            for q, s in zip(dataset[question_name], dataset[source_name])
        ]
        dataset = dataset[~np.array(already_generated)]
    document_ids = dataset["document_id"].tolist()
    source = dataset[source_name].tolist()
    question = dataset[question_name].tolist()
    return document_ids, source, question


def get_data_generic(cfg, run_name):
    ds_name = cfg["dataset"]["name"]
    source_name = cfg["dataset"]["source_name"]
    if ds_name in ["cnn_dailymail"]:
        split_name = cfg["dataset"]["split"]
        dataset = datasets.load_dataset(
            ds_name, cfg["dataset"]["version"], cache_dir=SCRATCH_CACHE_DIR
        )
        if split_name in ["train", "validation", "test"]:
            document_ids = [i for i in range(len(dataset[split_name][source_name]))]
            source = dataset[split_name][source_name]
        else:
            raise ValueError(
                f"Invalid or not implemented split {split_name} (e.g., all)"
            )
    elif ds_name in ["covidet", "debatepedia", "multioped", "qmsum", "duc_single"]:
        ds_path = cfg["dataset"]["path"]
        document_ids_name = cfg["dataset"]["document_id_name"]
        dataset = pd.read_csv(ds_path)
        duplicated_documents = dataset[source_name].duplicated(keep="first")
        dataset = dataset[~duplicated_documents]
        if "generated_jsonl_paths" in cfg["generation"]:
            generated_jsonlines = []
            for jsonlines_path in cfg["generation"]["generated_jsonl_paths"]:
                generated_jsonlines += read_jsonlines(jsonlines_path)
            # Write existing to the jsonl file
            write_to_jsonlines = []
            for elt in generated_jsonlines:
                if elt not in write_to_jsonlines:
                    write_to_jsonlines.append(elt)
            append_jsonlines(
                write_to_jsonlines,
                output_directory=cfg["output_directory"],
                run_name=run_name,
            )
            # Only keep documents that do not have generation
            existing_documents = list(
                set([elt["source"] for elt in generated_jsonlines])
            )
            dataset = dataset[~dataset[source_name].isin(existing_documents)]
        document_ids = dataset[document_ids_name].tolist()
        source = dataset[source_name].tolist()
    else:
        raise ValueError(f"Invalid dataset {ds_name}")
    return document_ids, source


def load_model(cfg):
    model_name = cfg["model"]["name"]
    tokenizer_name = cfg["tokenizer"]["name"]
    if tokenizer_name == model_name and model_name.startswith("t5"):
        # See https://github.com/huggingface/transformers/pull/24565 for why legacy=False
        tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_name, cache_dir=SCRATCH_CACHE_DIR, legacy=False
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=SCRATCH_CACHE_DIR
        ).to("cuda")
    elif tokenizer_name == model_name == "google/flan-t5-large":
        tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_name, cache_dir=SCRATCH_CACHE_DIR, legacy=False
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=SCRATCH_CACHE_DIR
        ).to("cuda")
    elif tokenizer_name == model_name == "facebook/bart-large-cnn":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=SCRATCH_CACHE_DIR
        )
        model = BartForConditionalGeneration.from_pretrained(
            model_name, cache_dir=SCRATCH_CACHE_DIR
        ).to("cuda")
    elif tokenizer_name == model_name == "google/pegasus-large":
        tokenizer = PegasusTokenizer.from_pretrained(
            tokenizer_name, cache_dir=SCRATCH_CACHE_DIR
        )
        model = (
            PegasusForConditionalGeneration.from_pretrained(
                model_name, cache_dir=SCRATCH_CACHE_DIR
            )
            .to("cuda")
            .half()
        )
    elif tokenizer_name == model_name == "allenai/led-large-16384-arxiv":
        tokenizer = LEDTokenizer.from_pretrained(
            tokenizer_name, cache_dir=SCRATCH_CACHE_DIR
        )
        model = (
            LEDForConditionalGeneration.from_pretrained(
                model_name, cache_dir=SCRATCH_CACHE_DIR
            )
            .to("cuda")
            .half()
        )
    elif tokenizer_name == model_name and "llama" in model_name:
        # FOllowing https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb#scrollTo=03c6e53e
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            padding_size="left",
            token=HF_TOKEN,
            cache_dir=SCRATCH_CACHE_DIR,
        )
        tokenizer.pad_token = "[PAD]"
        tokenizer.model_max_length = cfg["tokenizer"]["maximum_length"]
        # Should use left padding since want the generation to immediately
        # start after the prompt
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
            token=HF_TOKEN,
            cache_dir=SCRATCH_CACHE_DIR,
        )
        model.bfloat16()
    else:
        raise ValueError(f"Invalid tokenizer {tokenizer_name} and model {model_name}")
    model.eval()
    return tokenizer, model


def tokenize_source(
    model_name,
    tokenizer,
    source,
    max_new_tokens=None,
):
    if "t5" in model_name:
        tokenized_source = tokenizer(
            ["summarize: " + s for s in source],
            padding="longest",  # Pad to longest sequence IN batch
            truncation=True,  # Truncate to max model length
            max_length=None,  # Default to max length of the model
            return_tensors="pt",
        )
    elif "llama" in model_name.lower():
        # Truncate the source to allow for the prompt formatting of s
        prompt = lambda s: (
            f"""
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant specializing in summarization. Provide the best summary you can.
        <</SYS>>
        Provide a summary for the following text:
        {s}
        Summary: [/INST]
        """
        )
        # Change padding, but only for now
        tokenizer.padding_side = "right"
        length_prompt = len(tokenizer(prompt("")).input_ids)
        tokenized_source = tokenizer(
            source,
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokenized_source = tokenized_source.input_ids
        print("Decoding tokenized source, this may take a while...")
        if (
            tokenized_source.shape[1] + length_prompt + max_new_tokens
        ) > tokenizer.model_max_length:
            truncated_source = tokenizer.batch_decode(
                tokenized_source[:, : -(length_prompt + max_new_tokens)],
                skip_special_tokens=True,
            )
        else:
            truncated_source = tokenizer.batch_decode(
                tokenized_source,
                skip_special_tokens=True,
            )
        print("Finished decoding!")
        tokenizer.padding_side = "left"
        # Padding has been retstore to left
        tokenized_source = tokenizer(
            [prompt(s) for s in truncated_source],
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
    else:
        tokenized_source = tokenizer(
            source,
            padding="longest",
            truncation=True,
            max_length=None,
            return_tensors="pt",
        )
    source_input_ids = tokenized_source.input_ids.to("cuda")
    attention_mask = tokenized_source.attention_mask.to("cuda")
    return source_input_ids, attention_mask


def tokenizer_source_and_question(model_name, tokenizer, question, source, cfg):
    assert len(source) == len(question)
    if "led" in model_name or "bart" in model_name:
        # Concatenate the latent and source
        # As https://arxiv.org/abs/2112.07637
        tokenized_source = tokenizer(
            [q + f" {tokenizer.sep_token} " + s for q, s in zip(question, source)],
            padding="longest",  # Pad to longest sequence IN batch
            truncation=True,  # Truncate to max model length
            max_length=None,  # Default to max length of the model
            return_tensors="pt",
        )
        # Convert multioped to a query
    elif "llama" in model_name.lower():
        # Truncate the source to allow for the prompt formatting of s
        prompt = cfg["prompt_template"]
        max_new_tokens = cfg["generation"]["generate_kwargs"]["max_new_tokens"]
        # Change padding, but only for now
        tokenizer.padding_side = "right"
        length_prompt = len(tokenizer(prompt.format(q="", s="")).input_ids)
        max_question_length = max([len(tokenizer(q).input_ids) for q in question])
        tokenized_source = tokenizer(
            source,
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokenized_source = tokenized_source.input_ids
        print("Decoding tokenized source, this may take a while...")
        if (
            tokenized_source.shape[1]
            + length_prompt
            + max_new_tokens
            + max_question_length
        ) > tokenizer.model_max_length:
            truncated_source = tokenizer.batch_decode(
                tokenized_source[
                    :, : -(length_prompt + max_new_tokens + max_question_length)
                ],
                skip_special_tokens=True,
            )
        else:
            truncated_source = tokenizer.batch_decode(
                tokenized_source,
                skip_special_tokens=True,
            )
        print("Finished decoding!")
        tokenizer.padding_side = "left"
        # Padding has been restored to left
        tokenized_source = tokenizer(
            [prompt.format(q=q, s=s) for q, s in zip(question, truncated_source)],
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
    source_input_ids = tokenized_source.input_ids.to("cuda")
    attention_mask = tokenized_source.attention_mask.to("cuda")
    return source_input_ids, attention_mask


def custom_generate(
    model, model_name, batch_source_input_ids, batch_attention_mask, cfg
):
    if model_name == "allenai/led-large-16384-arxiv":
        # put global attention on <s> token as recommended in
        # https://huggingface.co/docs/transformers/model_doc/led#usage-tips
        global_attention_mask = torch.zeros_like(batch_attention_mask)
        global_attention_mask[:, 0] = 1
        with torch.no_grad():
            outputs = model.generate(
                batch_source_input_ids,
                attention_mask=batch_attention_mask,
                global_attention_mask=global_attention_mask,
                **cfg["generation"]["generate_kwargs"],
            )
    else:
        with torch.no_grad():
            outputs = model.generate(
                batch_source_input_ids,
                attention_mask=batch_attention_mask,
                **cfg["generation"]["generate_kwargs"],
            )
    return outputs["sequences"].to("cpu"), outputs["sequences_scores"].to("cpu")


def write_batch(
    batch_number,
    batch_size,
    num_batches,
    num_summaries,
    source,
    document_ids,
    output_seqs,
    output_seq_scores,
    tokenizer,
    run_name,
    cfg,
    question=None,
):
    assert batch_number == (num_batches - 1) or output_seqs.shape[0] == (
        batch_size * num_summaries
    )
    # Decode and calculate scores
    source_batch = [
        s
        for s in source[batch_number * batch_size : (batch_number + 1) * batch_size]
        for _ in range(num_summaries)
    ]
    if question is not None:
        question_batch = [
            q
            for q in question[
                batch_number * batch_size : (batch_number + 1) * batch_size
            ]
            for _ in range(num_summaries)
        ]
    document_id_batch = [
        d
        for d in document_ids[
            batch_number * batch_size : (batch_number + 1) * batch_size
        ]
        for _ in range(num_summaries)
    ]
    predictions_batch = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
    assert (
        len(source_batch)
        == len(document_id_batch)
        == len(predictions_batch)
        == len(output_seq_scores)
    )
    if question is None:
        dict_to_write = [
            {
                "document_id": document_id_batch[batch_number],
                "source": source_batch[batch_number],
                "pred": predictions_batch[batch_number],
                "pred_score": output_seq_scores[batch_number].item(),
            }
            for batch_number in range(len(source_batch))
        ]
    else:
        dict_to_write = [
            {
                "document_id": document_id_batch[batch_number],
                "source": source_batch[batch_number],
                "pred": predictions_batch[batch_number],
                "pred_score": output_seq_scores[batch_number].item(),
                "question": question_batch[batch_number],
            }
            for batch_number in range(len(source_batch))
        ]
    append_jsonlines(
        dict_to_write, output_directory=cfg["output_directory"], run_name=run_name
    )


def generate_generic_summary(run_name, cfg):
    # Load source reference and document ids with duplication
    document_ids, source = get_data_generic(cfg, run_name)
    num_summaries = cfg["generation"]["generate_kwargs"]["num_return_sequences"]
    # Load model
    model_name = cfg["model"]["name"]
    tokenizer, model = load_model(cfg)
    # Tokenize source
    source_input_ids, attention_mask = tokenize_source(
        model_name=model_name,
        tokenizer=tokenizer,
        source=source,
        max_new_tokens=cfg["generation"]["generate_kwargs"]["max_new_tokens"],
    )
    # Generate by batch
    batch_size = cfg["generation"]["batch_size"]
    num_batches = source_input_ids.shape[0] // batch_size + (
        1 if source_input_ids.shape[0] % batch_size != 0 else 0
    )
    # Generate summaries
    for i in tqdm(range(num_batches)):
        # Generate
        batch_source_input_ids = source_input_ids[i * batch_size : (i + 1) * batch_size]
        batch_attention_mask = attention_mask[i * batch_size : (i + 1) * batch_size]
        output_seqs, output_seq_scores = custom_generate(
            model=model,
            model_name=model_name,
            batch_source_input_ids=batch_source_input_ids,
            batch_attention_mask=batch_attention_mask,
            cfg=cfg,
        )
        write_batch(
            batch_number=i,
            batch_size=batch_size,
            num_batches=num_batches,
            num_summaries=num_summaries,
            source=source,
            document_ids=document_ids,
            output_seqs=output_seqs,
            output_seq_scores=output_seq_scores,
            tokenizer=tokenizer,
            run_name=run_name,
            cfg=cfg,
        )


def generate_e2e_summary(run_name, cfg):
    # Load source reference and document ids with duplication
    document_ids, source, question = get_data_e2e(cfg, run_name)
    num_summaries = cfg["generation"]["generate_kwargs"]["num_return_sequences"]
    # Load model
    model_name = cfg["model"]["name"]
    tokenizer, model = load_model(cfg)
    # Tokenize source
    source_input_ids, attention_mask = tokenizer_source_and_question(
        model_name=model_name,
        tokenizer=tokenizer,
        question=question,
        source=source,
        cfg=cfg,
    )
    # Generate by batch
    batch_size = cfg["generation"]["batch_size"]
    num_batches = source_input_ids.shape[0] // batch_size + (
        1 if source_input_ids.shape[0] % batch_size != 0 else 0
    )
    # Generate summaries
    for i in tqdm(range(num_batches)):
        # Generate
        batch_source_input_ids = source_input_ids[i * batch_size : (i + 1) * batch_size]
        batch_attention_mask = attention_mask[i * batch_size : (i + 1) * batch_size]
        output_seqs, output_seq_scores = custom_generate(
            model=model,
            model_name=model_name,
            batch_source_input_ids=batch_source_input_ids,
            batch_attention_mask=batch_attention_mask,
            cfg=cfg,
        )
        write_batch(
            batch_number=i,
            batch_size=batch_size,
            num_batches=num_batches,
            num_summaries=num_summaries,
            source=source,
            question=question,
            document_ids=document_ids,
            output_seqs=output_seqs,
            output_seq_scores=output_seq_scores,
            tokenizer=tokenizer,
            run_name=run_name,
            cfg=cfg,
        )


def get_lead_predictions(sources, n_sentences):
    lead_sentences = []
    nlp = stanza.Pipeline(lang="en", processors="tokenize")
    for source in tqdm(sources):
        doc = nlp(source)
        sentences = [
            " ".join([token.text for token in sent.tokens]) for sent in doc.sentences
        ]
        lead_sentences += [" ".join(sentences[:n_sentences])]
    return lead_sentences


def get_sentences(documents):
    nlp = stanza.Pipeline(lang="en", processors="tokenize")
    sentences = []
    for document in documents:
        doc = nlp(document)
        sentences += [
            [" ".join([token.text for token in sent.tokens]) for sent in doc.sentences]
        ]
    return sentences


def search_optimal_sentence(source_sents, reference, greedy_size):
    rouge_scores = compute_rouge(
        predictions=source_sents,
        references=[reference] * len(source_sents),
    )
    rouge_L = rouge_scores["rougeL"]
    # Gets top-k values -> indices
    if len(rouge_L) < greedy_size:
        return np.arange(len(rouge_L))
    else:
        indices = np.argpartition(rouge_L, -greedy_size)[-greedy_size:]
    return np.sort(indices)


def generate_max_sent_rouge(source_sents, ref_sents, greedy_size):
    pred_summary = []
    reference = " ".join(ref_sents)
    indices = search_optimal_sentence(source_sents, reference, greedy_size)
    return [source_sents[idx] for idx in indices]


def get_max_sent_rouge(document_ids, sources, refs, greedy_size, run_name, cfg):
    source_sentences = get_sentences(sources)
    ref_sentences = get_sentences(refs)
    for doc_id, source_sents, ref_sents in tqdm(
        zip(document_ids, source_sentences, ref_sentences)
    ):
        pred_summary = generate_max_sent_rouge(source_sents, ref_sents, greedy_size)
        pred_summary = " ".join(pred_summary)
        append_jsonlines(
            [
                {
                    "document_id": doc_id,
                    "source": " ".join(source_sents),
                    "pred": pred_summary,
                }
            ],
            output_directory=cfg["output_directory"],
            run_name=run_name,
        )


def generate_baseline_summary(run_name, cfg):
    # Get data
    raw_data = pd.read_csv(cfg["dataset"]["path"])
    document_ids = raw_data[cfg["dataset"]["document_id_name"]].tolist()
    sources = raw_data[cfg["dataset"]["source_name"]]
    assert len(document_ids) == len(sources)
    if cfg["summary_type"] == "lead":
        preds = get_lead_predictions(sources=sources, n_sentences=cfg["n_sentences"])
        assert len(preds) == len(sources)
        append_jsonlines(
            [
                {"document_id": doc_id, "source": source, "pred": pred}
                for doc_id, source, pred in zip(document_ids, sources, preds)
            ],
            output_directory=cfg["output_directory"],
            run_name=run_name,
        )
    elif cfg["summary_type"] == "max_sentence_rouge":
        refs = raw_data[cfg["dataset"]["reference_name"]]
        get_max_sent_rouge(
            document_ids, sources, refs, cfg["greedy_size"], run_name=run_name, cfg=cfg
        )


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "generate_summaries", "conf"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig) -> None:
    """
    Run with something like:
    python summary_generations.py --config-dir=conf/t5_generic --config-name=config1
    """
    # Set seed if specified
    if cfg.get("generation") and "seed" in cfg.get("generation"):
        set_seed(cfg["generation"]["seed"])
    if (
        "generation" in cfg
        and "summary_type" in cfg["generation"]
        and cfg["generation"]["summary_type"] == "generic"
    ):
        generate_generic_summary(run_name, cfg)
    elif (
        "generation" in cfg
        and "summary_type" in cfg["generation"]
        and cfg["generation"]["summary_type"] == "e2e"
    ):
        generate_e2e_summary(run_name, cfg)
    elif "summary_type" in cfg and cfg["summary_type"] == "lead":
        generate_baseline_summary(run_name, cfg)
    elif "summary_type" in cfg and cfg["summary_type"] == "max_sentence_rouge":
        generate_baseline_summary(run_name, cfg)


if __name__ == "__main__":
    main()
