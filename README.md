# RSASumm

Can we use RSA for Summ?


# Outline of work

1. Find datasets
- DUC: Specifically, the years with query-focused summarization (2006/2007).
- TAC: Same comment as above.
- Debatepedia (2017):
    - Description: Extracted from Debatepedia a list of triples of D-Q-S.
    - Link: https://github.com/PrekshaNema25/DiverstiyBasedAttentionMechanism/tree/master
    - Comment: It appears that some of the tiples in this dataset are of low quality. This paper https://arxiv.org/pdf/2305.06147.pdf tries to fix that by using ChatGPT to filter incorrect triples.
- MARGE (2021):
    - Link: https://github.com/yumoxu/marge, requires regeneration.
- MA-News (2019)
    - Link: https://github.com/ColiLea/aspect_based_summarization, but ask you to regenerate it.
- AQuaMuse (2020)
    - Description: Automatically generated dataset where a subset of (question/long answer) pairs from Google's Open Domain QA dataset are matched with *multiple* documents scraped from common crawl web repository.
- QMSum (2021)
    - Description: Query-based meeting summarization dataset. The input document is the whole meeting transcript then there are annotated generic summaries as well as annotated topic-based summaries. Importantly, there are also the annotations of which portions of the transcript are useful for summarizing the transcript for some particular topic.
    - Link: https://github.com/Yale-LILY/QMSum
- WikiAsp (2021)
    - Description: Similar to OASum in that it is automatically generated aspect-based summarization dataset using Wikipedia articles. In this case, they take each section of a Wikipedia article, filter it down and use the documents that are cited by that section as the source documents.
    - Link: https://github.com/neulab/wikiasp
- MultiOpEd (2021)
    - Description: This dataset is scraped from an editorial board with two constrasting takes to a given query. For each query, there are 2 editorials with different opinions, 2 abstracts (one for each editorial) and 2 high-level takeways (which the authors call *perspectives*).
    - Comments: Articles are in general pretty long but this seems to be a dataset that aligns with what you want to do.
    - Link: https://github.com/CogComp/MultiOpEd
- NEWSTS (2022)
    - Description: Annotate a subset of CNN/DM by using MTurkers to create two reference summaries for each article, each focused on some topic.
    - Link: https://aclanthology.org/2022.findings-acl.42.pdf
- CovidET (2022)
    - Description: Given a Covid-19-related Reddit post, annotators are asked to summarize them with a particular emotion in mind.
    - Comments: Reddit posts are relatively short, so could fit within context-window.
    - Link: https://github.com/honglizhan/CovidET
- SQuALITY (2022)
    - Description: Given a short-story ask **human annotators** to write summaries of the short-story. They get 1 general summary and 4 query-focused summaries. 
    - Comments: They note that automatic evaluation metrics correlate poorly with human judgements. Also, the input documents are quite long so should probably not start with these documents if want to test pre-trained LLMs only (they only explored fine-tuning).
    - Link: https://github.com/nyu-mll/SQuALITY
- OASum (2023)
    - Description: Given a Wikipedia page, the abstract of the Wikiepedia page can be seen as a multi-aspect summary of the entire Wikipedia article where one aspect relates to the topic of a section. The soruce document(s) (from one Wikipedia page) are the subsections and the reference summary is the filtered down abstract based on some ROUGE-based heuristic on its similarity to that section.
    - Comments: These summaries are still ill-defined but "less" than generic summaries. This is because the aspect for the summary is still broad in many cases. Also, most source documents are very long and are Wikipedia-based so Llama-2 will probably not be able to handle the length + it will have memorized the data.
    - Link: https://huggingface.co/datasets/kqsong/OASum?row=22
    - Potential experiments: Do summarization (Using fine-tuned model? Longformer, might be a pain to get working.) and then re-rank candidate summaries based on something like "What aspects/topics does this summary cover?"
- LMGQS
    - Comments: No link to a dataset. Not usable.

2. Find currently used models
- BART:
    - Comments: BART-Large has a max context-window of 1024.
    - Requires fine-tuning?: Probably.
- BART+DPR:
    - Comments: First retrieves sentences that are most useful to the query.
    - Requires fine-tuning?: Probably.
- PEGASUS:
    - Comments: Specifically fine-tuned for summarization.
    - Requires fine-tuning?: Probably.
- LED:
    - Comments: Longformer as encoder and transformer as decoder.
- T5:
    - Comments: Already used it in a preliminary experiment with news summarization.
3. Run inference-only experiments first with e.g. Llama-2
- This is lower risk/difficulty
- Priority:
    1. Run experiments where source document fits in Llama-2 and other pre-trained summarization systems. Do this only in the output space.
4. Run fine-tuning experiments second
- This is higher risk/difficulty