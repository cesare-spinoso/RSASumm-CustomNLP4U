import os

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.utils.dataset import get_question_column_name


class LiteralSummarizerDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        path_to_csv: str,
        tokenizer: AutoTokenizer,
        dataset_formatting=None,
    ):
        # Initialize
        self.dataset_name = dataset_name
        self.path_to_csv = path_to_csv
        self.tokenizer = tokenizer
        self.dataset_formatting = dataset_formatting
        self.is_built = False
        # Set new variables
        self.data = pd.read_csv(self.path_to_csv)
        self.source_name = "document"
        self.ref_summary_name = "summary"
        self.info_req_name = get_question_column_name(self.dataset_name)
        if self.dataset_name == "covidet":
            info_req_formatter = self.dataset_formatting["info_req_formatter"]
            self.data[self.info_req_name] = self.data["emotion"].apply(
                lambda emotion: info_req_formatter.format(emotion)
            )

    def build(self):
        if self.dataset_formatting["data_format"] == "e2e":
            self.build_e2e()
        self.is_built = True

    def build_e2e(self):
        # Tokenizer the input
        source = self.data[self.source_name].tolist()
        info_req = self.data[self.info_req_name].tolist()
        self.tokenized_input = self.tokenizer(
            [
                ir + f" {self.tokenizer.sep_token} " + s
                for ir, s in zip(info_req, source)
            ],
            padding="longest",  # Pad to longest sequence IN batch
            truncation=True,  # Truncate to max model length
            max_length=None,  # Default to max length of the model
            return_tensors="pt",
        )
        reference_summary = self.data[self.ref_summary_name].tolist()
        self.tokenized_label = self.tokenizer(
            reference_summary,
            padding="longest",  # Pad to longest sequence IN batch
            truncation=True,  # Truncate to max model length
            max_length=None,  # Default to max length of the model
            return_tensors="pt",
        )
        assert len(self.tokenized_input["input_ids"]) == len(
            self.tokenized_label["input_ids"]
        )

    def __len__(self):
        if not self.is_built:
            raise ValueError("Dataset is not built yet")
        return len(self.tokenized_input["input_ids"])

    def _decode(self, tokenized):
        return self.tokenizer.decode(tokenized, skip_special_tokens=True)

    def __getitem__(self, index, human_readable=False):
        if not self.is_built:
            raise ValueError("Dataset is not built yet")
        # Index
        source_ids = self.tokenized_input["input_ids"][index]
        source_mask = self.tokenized_input["attention_mask"][index]
        label_ids = self.tokenized_label["input_ids"][index]
        label_mask = self.tokenized_label["attention_mask"][index]
        # Output in human readable or not
        if not human_readable:
            return {
                "source_ids": source_ids,
                "source_mask": source_mask,
                "label_ids": label_ids,
                "label_mask": label_mask,
            }
        else:
            return {
                "source": self._decode(source_ids),
                "label": self._decode(label_ids),
            }


class LiteralSummarizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        csv_dir: str,
        tokenizer: AutoTokenizer,
        dataset_formatting=None,
        train_batch_size: int = 8,
        test_batch_size: int = 8,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.csv_dir = csv_dir
        self.tokenizer = tokenizer
        self.dataset_formatting = dataset_formatting
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def prepare_data(self) -> None:
        self.data_dict = dict.fromkeys(["train", "val"])
        for split in self.data_dict.keys():
            path_to_csv = os.path.join(self.csv_dir, f"{split}.csv")
            self.data_dict[split] = LiteralSummarizerDataset(
                dataset_name=self.dataset_name,
                path_to_csv=path_to_csv,
                tokenizer=self.tokenizer,
                dataset_formatting=self.dataset_formatting,
            )

    def setup(self, stage=None) -> None:
        if stage == "fit":
            for split in self.data_dict.keys():
                self.data_dict[split].build()
        else:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.data_dict["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_dict["val"],
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=1,
        )
