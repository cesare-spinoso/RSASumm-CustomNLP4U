import os

import hydra
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from src import HF_TOKEN, SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines, read_jsonlines


def load_model(cfg):
    model_name = cfg["model"]["name"]
    tokenizer_name = cfg["tokenizer"]["name"]
    if tokenizer_name == model_name and "Llama-3" in model_name:
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
    else:
        raise ValueError(f"Invalid tokenizer {tokenizer_name} and model {model_name}")
    model.eval()
    return tokenizer, model


def tokenize_qgen(model_name, prompt, num_questions, tokenizer, document, cfg):
    assert "llama" in model_name.lower()
    # Change padding, but only for now
    length_prompt = len(tokenizer(prompt.format(n="10", d="")).input_ids)
    max_new_tokens = cfg["qgen"]["generate_kwargs"]["max_new_tokens"]
    tokenized_document = (
        tokenizer(
            document,
            padding="longest",
            truncation=True,
            max_length=(
                tokenizer.model_max_length
                - length_prompt
                - max_new_tokens
                - cfg["qgen"].get("buffer_length", 0)
            ),
        )
    ).input_ids
    truncated_document = tokenizer.batch_decode(
        tokenized_document, skip_special_tokens=True
    )
    tokenized_document = tokenizer(
        [prompt.format(n=num_questions, d=d) for d in truncated_document],
        padding="longest",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    return tokenized_document.input_ids.to(
        "cuda"
    ), tokenized_document.attention_mask.to("cuda")


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "generate_questions", "conf"),
)
@main_decorator
def main(run_name, cfg):
    # Load model and tokenizer
    model_name = cfg["model"]["name"]
    tokenizer, model = load_model(cfg)
    # Load preprocessed dataset
    df = pd.read_csv(cfg["dataset"]["preprocessed_dataset_path"])
    if "generated_questions_path" in cfg["qgen"]:
        generated_questions_path = cfg["qgen"]["generated_questions_path"]
        generated_questions = read_jsonlines(generated_questions_path)
        if "empty_questions_path" in cfg["qgen"]:
            df_empty_questions = pd.read_csv(cfg["qgen"]["empty_questions_path"])
            documents_with_no_questions = df_empty_questions["document"].values.tolist()
            generated_questions = [elt for elt in generated_questions if elt["document"] not in documents_with_no_questions]
        append_jsonlines(generated_questions, cfg["output_directory"], run_name)
        generated_documents = [elt["document"] for elt in generated_questions]
        already_generated = [doc in generated_documents for doc in df["document"]]
        df = df[~np.array(already_generated)]
    # Fetch the document id, document and question
    # Removing any duplicates
    assert (~df["document"].isna()).sum() > 0 and (~df["question"].isna()).sum() > 0
    duplicated = df["document"].duplicated(keep="first")
    df_new = df[~duplicated]
    # Tokenize for QAing
    tokenized_input_ids, attention_mask = tokenize_qgen(
        model_name=model_name,
        tokenizer=tokenizer,
        num_questions=cfg["qgen"]["num_questions"],
        prompt=cfg["qgen"]["prompt"],
        document=df_new["document"].values.tolist(),
        cfg=cfg,
    )
    document_ids = df_new["document_id"].values.tolist()
    documents = df_new["document"].values.tolist()
    batch_size = cfg["qgen"]["batch_size"]
    num_batches = len(document_ids) // batch_size + (
        1 if len(document_ids) % batch_size != 0 else 0
    )
    notify_size = cfg["dataset"].get("notify_size")
    for i in tqdm(range(num_batches)):
        if notify_size is not None and (i + 1) * batch_size > notify_size:
            print(f"Processed > {notify_size} documents")
            notify_size = None
        s = slice(i * batch_size, (i + 1) * batch_size)
        batch_doc_ids = document_ids[s]
        batch_docs = documents[s]
        batch_input_ids = tokenized_input_ids[s]
        batch_attention_mask = attention_mask[s]
        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                **cfg["qgen"]["generate_kwargs"],
            )
        generated_questions = tokenizer.batch_decode(
            outputs["sequences"].to("cpu"),
            skip_special_tokens=True,
        )
        # Write to new jsonl
        dict_to_write = [
            {
                "document_id": batch_doc_ids[i],
                "document": batch_docs[i],
                "generated_answer": generated_questions[i],
            }
            for i in range(len(batch_doc_ids))
        ]
        append_jsonlines(dict_to_write, cfg["output_directory"], run_name)


if __name__ == "__main__":
    main()
