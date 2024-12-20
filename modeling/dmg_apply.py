from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from confidenceinterval import accuracy_score, classification_report_with_ci
from confidenceinterval.bootstrap import bootstrap_ci

class modelApply:
    def __init__(self, data_path) -> None:
        self.data_path = Path(data_path)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        self.hpt_experiment_name = f"{str(self.data_path.stem)}_hyperparameter"
        self.experiment_name = f"{str(self.data_path.stem)}_testing"
        maximum_mlflow_runs = self.get_maximum_mlflow()
        for max_run in maximum_mlflow_runs:
            run_id = max_run[1]
            geo_weighting = max_run[7]
            walk_strategy = max_run[8]
            walk_distance = max_run[9]
            num_walks = max_run[10]
            self.model_path = Path(f"model/{str(self.data_path.stem)}/{geo_weighting}_{walk_strategy}_all_{walk_distance}_{num_walks}/word2vec.model")
            _, _, test_set = self.data_preprocess()
            self.apply_and_test_model(run_id, geo_weighting, walk_strategy, walk_distance, num_walks, test_set)


    def prepare_vector_data(self, dataset_path, word2vec_path):
        nodes = pd.read_csv(f"{self.data_path}/nodes.int.csv")
        nodes = nodes[["index", "label"]]
        dataset = pd.read_csv(dataset_path)
        dataset = pd.merge(nodes, dataset, how="inner", on="index")
        vector_model = Word2Vec.load(str(word2vec_path))
        node_labels = dataset["label"].tolist()
        node_vector_list = [vector_model.wv[node_label] for node_label in node_labels]
        vector_data = pd.DataFrame(node_vector_list)
        vector_data = vector_data.add_prefix("vector_", axis=1)
        vector_data["label"] = node_labels
        prepared_data = pd.merge(dataset, vector_data, how="inner", on="label")
        prepared_data = prepared_data.drop(["index", "label"], axis=1)
        label = prepared_data.pop("class")
        return prepared_data, label

    def data_preprocess(self) -> tuple:
        print('Data Preprocessing')
        train_data, train_label = self.prepare_vector_data(f"{self.data_path}/training.int.csv", self.model_path)
        validation_data, validation_label = self.prepare_vector_data(f"{self.data_path}/validation.int.csv", self.model_path)
        test_data, test_label = self.prepare_vector_data(f"{self.data_path}/testing.int.csv", self.model_path)
        # return trainData and testdata X and Y dataframe in tuple format
        return (train_data, train_label), (validation_data, validation_label), (test_data, test_label)

    def get_maximum_mlflow(self) -> np.recarray:
        # Initialize the MLflow client
        client = MlflowClient(tracking_uri="sqlite:///data/mlruns.db")
        experiment = client.search_experiments(filter_string=f"name = '{self.hpt_experiment_name}'")
        experiment_id = experiment.to_list()[0].experiment_id
        runs = client.search_runs(experiment_ids=[experiment_id], max_results=50000)

        # Check each run's tags to see if it has a parentRunId (indicating a child run)
        run_list = []
        for run in tqdm(runs, desc="MLflow iteration"):
            run_json = {}
            parent_run_id = run.data.tags.get("mlflow.parentRunId")
            if parent_run_id:
                run_json["parent_run_id"] = parent_run_id
                parent_run_details = client.get_run(parent_run_id)
                run_details = run.to_dictionary()
                run_id = run_details.get("info").get("run_id")
                run_json["run_id"] = run_id
                run_details_metrics = run_details.get("data").get("metrics")
                run_json.update(run_details_metrics)
                parent_run_details = parent_run_details.to_dictionary()
                parent_run_parameters = parent_run_details.get("data").get("params")
                run_json.update(parent_run_parameters)
                run_list.append(run_json)
            else:
                continue

        run_data_df = pd.DataFrame(run_list)
        max_parent_run_index = run_data_df.groupby(by="parent_run_id")["mcc"].idxmax()
        selected_idx = max_parent_run_index.to_list()
        max_run_data_df = run_data_df.filter(items=selected_idx, axis=0)
        max_run_array = max_run_data_df.to_records(index=False)
        return max_run_array

    def apply_and_test_model(self, run_id: str, geo_weighting: bool, walk_strategy: str, walk_distance: int, num_walks: int, test_set):
        test_data, test_label = test_set
        mlflow.set_experiment(self.hpt_experiment_name)
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            mlflow.log_params({"geo_weighting": geo_weighting, "walk_strategy": walk_strategy, "walk_distance": walk_distance, "num_walks": num_walks})
            prediction = loaded_model.predict(test_data)
            accuracy, accuracy_interval = accuracy_score(test_label, prediction)
            low_accuracy, high_accuracy = accuracy_interval
            metric_report = classification_report_with_ci(test_label, prediction)
            dict_metric_report = metric_report[metric_report["Class"] == "macro"].to_dict("list")
            precision = dict_metric_report["Precision"][0]
            precision_low, precision_high = dict_metric_report["Precision CI"][0]
            recall = dict_metric_report["Recall"][0]
            recall_low, recall_high = dict_metric_report["Recall CI"][0]
            f1_score = dict_metric_report["F1-Score"][0]
            f1_score_low, f1_score_high = dict_metric_report["F1-Score CI"][0]
            mcc_score, mcc_ci = bootstrap_ci(test_label, prediction, metric=matthews_corrcoef, random_state=42)
            mcc_score_low, mcc_score_high = mcc_ci
            mlflow.log_metrics({"accuracy": accuracy, "accuracy_lower_95": low_accuracy, "accuracy_higher_95": high_accuracy,
                                "precision": precision, "precision_lower_95": precision_low, "precision_higher_95": precision_high,
                                "recall": recall, "recall_lower_95": recall_low, "recall_higher_95": recall_high,
                                "F1": f1_score, "f1_lower_95": f1_score_low, "f1_higher_95": f1_score_high,
                                "MCC": mcc_score, "mcc_lower_95": mcc_score_low, "mcc_higher_95": mcc_score_high})

if __name__ == "__main__":
    folder_path = "data/kgbench_dmg777k"
    modelApply(folder_path)
