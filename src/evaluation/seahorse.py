# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src import SCRATCH_CACHE_DIR
import pandas as pd

from src.utils.helper import read_jsonlines

tokenizer = AutoTokenizer.from_pretrained(
    "google/seahorse-large-q1",
    cache_dir=SCRATCH_CACHE_DIR,
)
model = AutoModelForSequenceClassification.from_pretrained("google/seahorse-large-q1", cache_dir=SCRATCH_CACHE_DIR,)

df_raw = pd.read_csv("/home/mila/c/cesare.spinoso/RSASumm/data/squality/test.csv")
df_summaries = pd.DataFrame(read_jsonlines("/home/mila/c/cesare.spinoso/RSASumm/data/generate_summaries/llama3_qfs/squality/2024-07-24-14:54:08.993934.jsonl"))

source_text = df_raw["document"].loc[0]
generated_summary = df_summaries["pred"].loc[0]
string_to_evaluate = f"premise: {source_text[:100]} hypothesis: {generated_summary[:100]}"
tokens_to_evaluate = tokenizer(string_to_evaluate, return_tensors="pt")
output = model(**tokens_to_evaluate)