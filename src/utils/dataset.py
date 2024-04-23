def get_question_column_name(dataset_name):
    if dataset_name in ["covidet", "duc_single", "multioped", "qmsum"]:
        return "question"
    elif dataset_name in ["debatepedia"]:
        return "query"
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
