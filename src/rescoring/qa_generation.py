import os
import warnings

from copy import deepcopy
import hydra
import numpy as np
import pandas as pd
import regex
from tqdm import tqdm
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from src import HF_TOKEN, SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines, read_jsonlines


def load_model(cfg):
    model_name = cfg["model"]["name"]
    tokenizer_name = cfg["tokenizer"]["name"]
    if tokenizer_name == model_name and "llama" in model_name:
        # Following https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb#scrollTo=03c6e53e
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_size="left",
            token=HF_TOKEN,
            cache_dir=SCRATCH_CACHE_DIR,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = cfg["tokenizer"]["model_max_length"]
        # Should use left padding since want the generation to immediately
        # start after the prompt
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            token=HF_TOKEN,
            cache_dir=SCRATCH_CACHE_DIR,
        )
        if "Llama-2" in model_name:
            tokenizer.pad_token = "[PAD]"
            model.bfloat16()
    else:
        raise ValueError(f"Invalid tokenizer {tokenizer_name} and model {model_name}")
    model.eval()
    return tokenizer, model


def tokenize_qa(model_name, prompt, tokenizer, question, source, cfg):
    assert "llama" in model_name.lower()
    assert len(question) == len(source)
    # Change padding, but only for now
    tokenizer.padding_side = "right"
    length_prompt = len(tokenizer(prompt.format(d="", q="")).input_ids)
    # NOTE: This order is intentional: Truncate the source not the question
    tokenized_source = (
        tokenizer(
            source,
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
    ).input_ids
    tokenized_question = (
        tokenizer(
            question,
            padding=False,
            truncation=False,
        )
    ).input_ids
    question_lengths = [len(q) for q in tokenized_question]
    print("Decoding tokenized source, this may take a while...")
    max_new_tokens = cfg["qa"]["generate_kwargs"]["max_new_tokens"]
    truncated_source = tokenizer.batch_decode(
        [
            tok_source[: -(length_prompt + max_new_tokens + q_length)]
            for tok_source, q_length in zip(tokenized_source, question_lengths)
        ],
        skip_special_tokens=True,
    )
    print("Finished decoding!")
    tokenizer.padding_side = "left"
    tokenized_source = tokenizer(
        [prompt.format(d=d, q=q) for d, q in zip(truncated_source, question)],
        padding="longest",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return tokenized_source.input_ids.to("cuda"), tokenized_source.attention_mask.to(
        "cuda"
    )


def preprocess_question_generation(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows_to_rerun = []
    rows_exploded = []
    warn = False
    for row in df.itertuples(index=False):
        q = row.question
        if "Questions: |eot_id|>assistantassistant" in q:
            q = q.split("Questions: |eot_id|>assistantassistant")[-1].strip()
        elif "Questions: |eot_id|>assistant" in q:
            q = q.split("Questions: |eot_id|>assistant")[-1].strip()
        qs = regex.findall(
            r"what[^\?]*\?|who[^\?]*\?|where[^\?]*\?|when[^\?]*\?|how[^\?]*\?|why[^\?]*\?",
            q,
            flags=regex.IGNORECASE,
        )
        if len(qs) == 0:
            warn = True
            rows_to_rerun.append(row._asdict())
            continue
        row = row._asdict()
        for split_q in qs:
            row = deepcopy(row)
            row["question"] = split_q
            rows_exploded.append(row)
    if warn:
        warnings.warn(
            f"At least one empty question generation was found, need to re-run for those documents. Writing them in a csv file empty_questions_{dataset_name}."
        )
        df_to_rerun = pd.DataFrame(rows_to_rerun)
        df_to_rerun.to_csv(f"empty_questions_{dataset_name}.csv", index=False)
    df_exploded = pd.DataFrame(rows_exploded)
    return df_exploded


def get_qa_data(run_name, cfg):
    if "preprocessed_dataset_path" in cfg["dataset"]:
        df = pd.read_csv(cfg["dataset"]["preprocessed_dataset_path"])
    elif "generated_qs_path" in cfg["dataset"]:
        jsonlines_data = read_jsonlines(cfg["dataset"]["generated_qs_path"])
        df = pd.DataFrame(jsonlines_data)
        df.rename(columns={"generated_answer": "question"}, inplace=True)
        df = preprocess_question_generation(dataset_name=cfg["dataset"]["name"], df=df)
    if "generated_jsonl_path" in cfg["qa"]:
        generated_jsonlines = read_jsonlines(cfg["qa"]["generated_jsonl_path"])
        append_jsonlines(
            generated_jsonlines,
            output_directory=cfg["output_directory"],
            run_name=run_name,
        )
        # Filter for existing generation
        questions = [elt["question"] for elt in generated_jsonlines]
        documents = [elt["source"] for elt in generated_jsonlines]
        generated_combined = list(zip(questions, documents))
        already_generated = [
            (q, s) in generated_combined for q, s in zip(df["question"], df["document"])
        ]
        df = df[~np.array(already_generated)]
    return df


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rescoring", "conf"),
)
@main_decorator
def main(run_name, cfg):
    # Load model and tokenizer
    model_name = cfg["model"]["name"]
    tokenizer, model = load_model(cfg)
    # Load preprocessed dataset
    df = get_qa_data(run_name, cfg)
    if "preset_question" in cfg["qa"]:
        df["question"] = cfg["qa"]["preset_question"]
    # Fetch the document id, document and question
    # Removing any duplicates
    assert (~df["document"].isna()).sum() > 0 and (~df["question"].isna()).sum() > 0
    df_new = df[["document_id", "document", "question"]].drop_duplicates()
    # Tokenize for QAing
    tokenized_input_ids, attention_mask = tokenize_qa(
        model_name=model_name,
        tokenizer=tokenizer,
        prompt=cfg["qa"]["prompt"],
        question=df_new["question"].values.tolist(),
        source=df_new["document"].values.tolist(),
        cfg=cfg,
    )
    document_ids = df_new["document_id"].values.tolist()
    documents = df_new["document"].values.tolist()
    questions = df_new["question"].values.tolist()
    batch_size = cfg["qa"]["batch_size"]
    if "sample_size" in cfg["dataset"]:
        num_batches = cfg["dataset"]["sample_size"] // batch_size + (
            1 if cfg["dataset"]["sample_size"] % batch_size != 0 else 0
        )
    else:
        num_batches = len(document_ids) // batch_size + (
            1 if len(document_ids) % batch_size != 0 else 0
        )
    for i in tqdm(range(num_batches)):
        s = slice(i * batch_size, (i + 1) * batch_size)
        batch_doc_ids = document_ids[s]
        batch_docs = documents[s]
        batch_questions = questions[s]
        batch_input_ids = tokenized_input_ids[s]
        batch_attention_mask = attention_mask[s]
        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                **cfg["qa"]["generate_kwargs"],
            )
        generated_answers = tokenizer.batch_decode(
            outputs["sequences"].to("cpu"),
            skip_special_tokens=True,
        )
        # Write to new jsonl
        dict_to_write = [
            {
                "document_id": batch_doc_ids[i],
                "source": batch_docs[i],
                "question": batch_questions[i],
                "generated_answer": generated_answers[i],
            }
            for i in range(batch_size)
        ]
        append_jsonlines(dict_to_write, cfg["output_directory"], run_name)


if __name__ == "__main__":
    main()
