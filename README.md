# RSA-QFS

Repository for the paper "Does This Summary Answer My Question? Modeling Query-Focused Summary Readers with Rational Speech Acts" presented as a poster at the [CustomNLP4U](https://customnlp4u-24.github.io/) Workshop at EMNLP 2024. The arxiv paper can be found [here](http://arxiv.org/abs/2411.06524). Poster can be found 

# Paper poster

![poster](https://github.com/cesare-spinoso/RSASumm-CustomNLP4U/blob/main/RSA-QFS%20CustomNLP4U%20Poster.png)

# Navigating the repository

- `src/data` contains the data preprocessing code.
- `src/finetuning/literal_summarizer` contains the fine-tuning code for the literal summarizer (i.e., BART).
- `src/generate_summaries` contains the code for generating the summaries (using both the fine-tuned summarizer as well as Llama3).
- `src/rescoring` contains the code for re-ranking the summaries based on the reader's ability to answer the question given the summary. It also contains the QA generation.
- `src/evaluation` contains the evaluation code for the pragmatic summarizer.

# Citation 

If you use our dataset, code, or findings, please cite us with
```
@misc{rsa-summ-customnlp4u,
      title={Does This Summary Answer My Question? Modeling Query-Focused Summary Readers with Rational Speech Acts}, 
      author={Cesare Spinoso-Di Piano and Jackie Chi Kit Cheung},
      year={2024},
      eprint={2411.06524},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2411.06524}, 
}
```
