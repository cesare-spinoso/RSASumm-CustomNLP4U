# Summary generation

## Generic summarizations

Description: Given the source and a summarization model, ask for 5 diverse summaries (using diverse decoding) recording the prediction scores as well.

Jsonlines file paths: Can be found in /RSASumm/src/generate_summaries/generated_summaries/generic_summaries.yaml
Information about 
- Flan-T5 Large: Max token limit 512. Uses a prompt "summarize: ".
    - Covidet:
    - Debatepedia:
    - DUC_single:
    - Multioped:
    - QMSum:
- BART: Max token limit 1024.
    - Covidet:
    - Debatepedia:
    - DUC_single:
    - Multioped:
    - QMSum:
- PEGASUS: Max token limit 1024 or 512 (unsure).
    - Covidet:
    - Debatepedia:
    - DUC_single:
    - Multioped:
    - QMSum:
- LED (Using Longformer): Max token limit is 16K tokens.
    - Covidet:
    - Debatepedia:
    - DUC_single:
    - Multioped:
    - QMSum:
- Llama2: Max token limit is 4K tokens. Also uses a prompt "summarize" and "summary". Llama2-7B-Chat seems to repeat the question/prompt before answering it which is surprising. So will need to extract it before you can use it.
    - Covidet:
    - Debatepedia:
    - DUC_single:
    - Multioped:
    - QMSum: