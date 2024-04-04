from src.utils.helper import read_yaml
from src.utils.visualizations import (
    convert_weighted_rsa_json_to_dfs,
    create_weighted_rsa_heatmap_figs,
    dropdown_heatmaps,
)


def main():
    yaml_file_path = "/home/mila/c/cesare.spinoso/RSASumm/src/evaluation/evaluations/rouge_scores_table.yaml"
    yaml_file_contents = read_yaml(yaml_file_path)

    fig_dict = {}
    for model_name, json_path in yaml_file_contents.items():
        if json_path is not None:
            dfs = convert_weighted_rsa_json_to_dfs(json_path)
            fig = create_weighted_rsa_heatmap_figs(dfs)
            fig_dict[model_name] = fig

    dash_app = dropdown_heatmaps(fig_dict, jupyter=False)
    return dash_app


if __name__ == "__main__":
    dash_app = main()
    dash_app.run_server(host="0.0.0.0", port=8080)
