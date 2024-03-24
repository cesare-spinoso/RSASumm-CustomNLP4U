import os

import hydra
import pandas as pd
from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from src import HF_TOKEN, SCRATCH_CACHE_DIR, SRC_DIRECTORY
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines


def load_model(cfg):
    model_name = cfg["model"]["name"]
    tokenizer_name = cfg["tokenizer"]["name"]
    if tokenizer_name == model_name and "llama" in model_name:
        # Following https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb#scrollTo=03c6e53e
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            padding_size="left",
            token=HF_TOKEN,
            cache_dir=SCRATCH_CACHE_DIR,
        )
        tokenizer.pad_token = "[PAD]"
        tokenizer.model_max_length = cfg["tokenizer"]["model_max_length"]
        # Should use left padding since want the generation to immediately
        # start after the prompt
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
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


def tokenize_source(model_name, tokenizer, question, source, cfg):
    assert "llama" in model_name.lower()
    assert len(question) == len(source)
    # Change padding, but only for now
    tokenizer.padding_side = "right"
    # Truncate the source to allow for the prompt formatting of s
    q_example = cfg["qa"]["incontext-question"]
    c_example = cfg["qa"]["incontext-content"]
    a_example = cfg["qa"]["incontext-answer"]
    prompt = (
        lambda q, c: f"""
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant specializing in question answering. Provide answers to questions given a context. For example,
    Question: {q_example}
    Context: {c_example}
    Answer: {a_example}
    <</SYS>>

    Question: {q}
    Context: {c}
    Answer: [/INST]
    """
    )
    length_prompt = len(tokenizer(prompt("", "")).input_ids)
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
    tokenized_source = tokenizer(
        [prompt(q, s) for q, s in zip(question, truncated_source)],
        padding="longest",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    tokenizer.padding_side = "left"
    return tokenized_source.input_ids, tokenized_source.attention_mask


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
    df = pd.read_csv(cfg["qa"]["preprocessed_dataset_path"])
    # Fetch the document id, document and question
    # Removing any duplicates
    assert (~df["document"].isna()).sum() > 0 and (~df["question"].isna()).sum() > 0
    df_new = df[["document_id", "document", "question"]].drop_duplicates()
    # Tokenize for QAing
    tokenized_input_ids, attention_mask = tokenize_source(
        model_name=model_name,
        tokenizer=tokenizer,
        question=df_new["question"].values.tolist(),
        source=df_new["document"].values.tolist(),
        cfg=cfg,
    )
    document_ids = df_new["document_id"].values.tolist()
    documents = df_new["document"].values.tolist()
    questions = df_new["question"].values.tolist()
    batch_size = cfg["qa"]["batch_size"]
    num_batches = len(document_ids) // batch_size
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
