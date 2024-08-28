from distutils.core import setup

# README file contents
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="RSA-Summ",
    version="0.1dev",
    packages=[
        "src",
    ],
    license="MIT",
    description="RSA Summ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cesare Spinoso-Di Piano",
    author_email="cesare.spinoso-dipiano@mail.mcgill.ca",
    install_requires=[
        "numpy",
        "pandas",
        "tabulate",
        "matplotlib",
        "scikit-learn",
        "nltk",
        "spacy",
        "gensim",
        "torch",
        "rank_bm25",
        "bs4",
        "plotly",
        "omegaconf",
        "hydra-core",
        "jsonlines",
        "transformers",
        "sentence-transformers",
        "tqdm",
        "seaborn",
        "scipy",
        "ipython",
        "protobuf",
        "evaluate",
        "stanza",
        "datasets",
        "sh",
        "accelerate",
        "bitsandbutes",
    ],
)
