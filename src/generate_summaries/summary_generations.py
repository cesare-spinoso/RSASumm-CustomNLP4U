import os

import datasets
import hydra
import pandas as pd
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
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines


def get_data_generic(cfg):
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
        source = dataset[~duplicated_documents][source_name].tolist()
        document_ids = dataset[~duplicated_documents][document_ids_name].tolist()
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
        # Change padding, but only for now
        tokenizer.padding_side = "right"
        # Truncate the source to allow for the prompt formatting of s
        prompt = (
            lambda s: f"""
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant specializing in summarization. Provide the best summary you can.
        <</SYS>>
        Provide a summary for the following text:
        {s}
        Summary: [/INST]
        """
        )
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
        truncated_source = tokenizer.batch_decode(
            tokenized_source[:, : -(length_prompt + max_new_tokens)],
            skip_special_tokens=True,
        )
        print("Finished decoding!")
        tokenized_source = tokenizer(
            [prompt(s) for s in truncated_source],
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        tokenizer.padding_side = "left"
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
    elif "llama" in model_name.lower():
        with torch.no_grad():
            outputs = model.generate(
                batch_source_input_ids,
                attention_mask=batch_attention_mask,
                **cfg["generation"]["generate_kwargs"],
            )
        return outputs["sequences"].to("cpu"), outputs["sequences_scores"].to("cpu")
    else:
        with torch.no_grad():
            outputs = model.generate(
                batch_source_input_ids,
                attention_mask=batch_attention_mask,
                **cfg["generation"]["generate_kwargs"],
            )
    return outputs["sequences"].to("cpu"), outputs["sequences_scores"].to("cpu")


def generate_generic_summary(run_name, cfg):
    # Load source reference and document ids with duplication
    document_ids, source = get_data_generic(cfg)
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
    num_batches = source_input_ids.shape[0] // batch_size
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
        assert i == (num_batches - 1) or output_seqs.shape[0] == (
            batch_size * num_summaries
        )
        # Decode and calculate scores
        source_batch = [
            s
            for s in source[i * batch_size : (i + 1) * batch_size]
            for _ in range(num_summaries)
        ]
        document_id_batch = [
            d
            for d in document_ids[i * batch_size : (i + 1) * batch_size]
            for _ in range(num_summaries)
        ]
        predictions_batch = tokenizer.batch_decode(
            output_seqs, skip_special_tokens=True
        )
        assert (
            len(source_batch)
            == len(document_id_batch)
            == len(predictions_batch)
            == len(output_seq_scores)
        )
        dict_to_write = [
            {
                "document_id": document_id_batch[i],
                "source": source_batch[i],
                "pred": predictions_batch[i],
                "pred_score": output_seq_scores[i].item(),
            }
            for i in range(len(source_batch))
        ]
        append_jsonlines(
            dict_to_write, output_directory=cfg["output_directory"], run_name=run_name
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
    if "seed" in cfg["generation"]:
        set_seed(cfg["generation"]["seed"])
    if cfg["generation"]["summary_type"] == "generic":
        generate_generic_summary(run_name, cfg)


if __name__ == "__main__":
    main()
