from vllm import LLM, SamplingParams
from src import SCRATCH_CACHE_DIR

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=SCRATCH_CACHE_DIR / "meta-llama--Meta-Lama-3-8B-Instruct")