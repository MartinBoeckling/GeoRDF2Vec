''''
Title:
Extra Gradient Boosting Script

Description:
This script uses the extra gradient boosting algorithm to predict wildfires. The
hyperparameters are optimized using bayes search cross validation implemented by 
the scikit-learn optimization package. The optimization is performed on a f1-score.
The datasets are highly imbalanced, therefore a random oversampling for the minority
class is performed. The random over sampling is implemented by the imblearn package.

Input:
    - dataPath: Path of dataset for use case in format dir/.../file
    - testDate: Date where split is performed

Output:
    - Optimal parameter combination
    - Score development over time
    - Classification report implemented by scikit-learn

'''
# import packages
import json
import argparse
from pathlib import Path
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from gensim.models import Word2Vec
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score
import mlflow



class modelPrediction:
    def __init__(self, data_path, resume, model_path, embedding_model):
        # add MLflow support
        # transform datafile path into pathlib object
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.embedding_model = embedding_model
        # create directory for use case
        rdf2vec_parameters = str(self.model_path.parent).split("/")[-1]
        self.logging_path = Path(f'logging/{str(self.data_path.stem)}').joinpath(rdf2vec_parameters)
        self.logging_path.mkdir(exist_ok=True, parents=True)
        self.resume = resume
        self.optimization_step = 0
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(f"{str(self.data_path.stem)}_normalized_hyperparameter")
        mlflow.xgboost.autolog()
        # perform data preprocessing for training and validation data
        train_set, validation_set, _ = self.data_preprocess()
        with mlflow.start_run(run_name=f'parent_run_{rdf2vec_parameters}') as run:
            self.parent_run_id = run.info.run_id
            if self.embedding_model == "RDF2Vec":
                rdf2vec_parameters = rdf2vec_parameters.split("_")
                self.rdf2vec_parameters = rdf2vec_parameters
                mlflow.log_params({'Geo Weighting': rdf2vec_parameters[0], "Walk strategy": rdf2vec_parameters[1], "max distance": rdf2vec_parameters[2], "number walks": rdf2vec_parameters[3]})
            else:
                mlflow.log_params({"Model": "TransGEO"})
            self.parameter_tuning(train_set, validation_set)
            # perform training based on train and test dataset and parametersettings
    

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

    def prepare_transgeo_data(self, dataset_path, embedding_path):
        dataset = pd.read_csv(dataset_path)
        entity_embedding_df = pd.read_csv(embedding_path)
        prepared_data = pd.merge(dataset, entity_embedding_df, on="index", how="inner")
        prepared_data = prepared_data.drop(["index"], axis=1)
        label = prepared_data.pop("class")
        return prepared_data, label

    def data_preprocess(self):
        print('Data Preprocessing')
        if self.embedding_model == "RDF2Vec":
            train_data, train_label = self.prepare_vector_data(f"{self.data_path}/training.int.csv", self.model_path)
            validation_data, validation_label = self.prepare_vector_data(f"{self.data_path}/validation.int.csv", self.model_path)
            test_data, test_label = self.prepare_vector_data(f"{self.data_path}/testing.int.csv", self.model_path)
        elif self.embedding_model == "TransGEO":
            train_data, train_label = self.prepare_transgeo_data(f"{self.data_path}/training.int.csv", self.model_path)
            validation_data, validation_label = self.prepare_transgeo_data(f"{self.data_path}/validation.int.csv", self.model_path)
            test_data, test_label = self.prepare_transgeo_data(f"{self.data_path}/testing.int.csv", self.model_path)
        else:
            raise NotImplementedError("Please provide a valid string with either RDF2Vec or TransGEO for the model name")
        
        # return trainData and testdata X and Y dataframe in tuple format
        return (train_data, train_label), (validation_data, validation_label), (test_data, test_label)


    def parameter_routine_cv(self, minChildWeight, maxDepth, subsample,
                          colsampleBytree, colsampleBylevel, gamma, numberEstimators):
        
        with mlflow.start_run(parent_run_id=self.parent_run_id, nested=True):
            xgb_classification = xgb.XGBClassifier(seed=15, n_jobs=20, tree_method='hist',
                                    min_child_weight=minChildWeight,
                                    max_depth = int(maxDepth),
                                    subsample=subsample,
                                    colsample_bytree = colsampleBytree,
                                    colsample_bylevel= colsampleBylevel,
                                    gamma = gamma,
                                    n_estimators = int(numberEstimators))
            
            xgb_classification.fit(self.train_data, self.train_label)
            prediction_label = xgb_classification.predict(self.validation_data)
            mcc_metric = matthews_corrcoef(self.validation_label, prediction_label)
            accuracy_metric = accuracy_score(self.validation_label, prediction_label)
            f1_metric = f1_score(self.validation_label, prediction_label, average="micro")
            precision_metric = precision_score(self.validation_label, prediction_label, average="micro")
            recall_metric = recall_score(self.validation_label, prediction_label, average="micro")
            if self.embedding_model == "RDF2Vec":
                mlflow.log_params({"Model": "RDF2Vec", 'Geo Weighting': self.rdf2vec_parameters[0], "Walk strategy": self.rdf2vec_parameters[1], "max distance": self.rdf2vec_parameters[3], "number walks": self.rdf2vec_parameters[4]})
            else:
                mlflow.log_params({"Model": "TransGEO"})
            mlflow.log_metrics({'mcc': mcc_metric, 'accuracy': accuracy_metric, 'Micro F1': f1_metric, 'Micro Precision': precision_metric, 'Micro Recall': recall_metric})
            mlflow.log_metrics(metrics={'mcc': mcc_metric, 'accuracy': accuracy_metric, 'Micro F1': f1_metric, 'Micro Precision': precision_metric, 'Micro Recall': recall_metric}, 
                               run_id=self.parent_run_id, step=self.optimization_step)
            self.optimization_step += 1
            return mcc_metric

    def parameter_tuning(self, train_set, validation_set):
        '''
        specify bayesian search cross validation with the following specifications
            - estimator: specified extra gradient boosting classifier
            - search_spaces: defined optimization area for hyperparameter tuning
            - cv: Specifying split for cross validation with 5 splits
            - scoring: optimization function based on f1_marco-score optimization
            - verbose: Output while optimizing
            - n_jobs: Parallel jobs to be used for optimization using 2 jobs
            - n_iter: Iteration for optimization
            - refit: Set to false as only parameter settings need to be extracted
        '''
        print('Parameter tuning')
        # extract dataframes from train and test data tuples
        self.train_data = train_set[0]
        self.train_label = train_set[1]
        self.validation_data = validation_set[0]
        self.validation_label = validation_set[1]
        optimizer = BayesianOptimization(f=self.parameter_routine_cv,
                                        pbounds={
                                            'minChildWeight': (0, 100),
                                            'maxDepth': (0, 50),
                                            'subsample': (0.01, 1.0),
                                            'colsampleBytree': (0.01, 1.0),
                                            'colsampleBylevel': (0.01, 1.0),
                                            'gamma': (0, 50),
                                            'numberEstimators': (50, 1000)
                                        },
                                        verbose=2,
                                        random_state=14)
        if self.resume:
            load_logs(optimizer, logs=[f'{self.logging_path}/logs.json'])
            with open(f'{self.logging_path}/logs.json', encoding="utf-8") as loggingFile:
                loggingFiles = [json.loads(jsonObj) for jsonObj in loggingFile]
            iterationSteps = 50 - len(loggingFiles) + 10
            initPoints = 0
        else:
            iterationSteps = 50
            initPoints = 10
        logger = JSONLogger(path=f"{self.logging_path}/logs.json")
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        optimizer.maximize(n_iter=iterationSteps, init_points=initPoints)
        print(f'Best parameter & score: {optimizer.max}')
        data_iteration_performance = pd.json_normalize(optimizer.res)
        data_iteration_performance.to_csv(f'{self.logging_path}/run_performance.csv', index=False)
        parameterNames = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'max_depth', 'min_child_weight', 'n_estimators', 'scale_pos_weight', 'subsample']
        parameterCombination = dict(zip(parameterNames, optimizer.max['params'].values()))
        parameterCombination['max_depth'] = int(parameterCombination['max_depth'])
        parameterCombination['n_estimators'] = int(parameterCombination['n_estimators'])

if __name__ == '__main__':
    
    pd.options.mode.chained_assignment = None  # default='warn'
    # initialize the command line argparser
    parser = argparse.ArgumentParser(description='XGBoost argument parameters')
    # add validation argument parser
    parser.add_argument('-r', '--resume', default=False, action='store_true',
    help="use parameter if grid parameter search should be resumed")
    # add path argument parser
    parser.add_argument('-p', '--path', type=str, required=True,
    help='string value to data path')
    parser.add_argument('-mp', '--model_path', type=str, required=True,
    help='string value to model path')
    parser.add_argument('-e', '--embedding_model', type=str, required=True,
    help='string value for embedding model')
    # store parser arguments in args variable
    args = parser.parse_args()
    # Pass arguments to class function to perform xgboosting
    model = modelPrediction(data_path=args.path, resume=args.resume, model_path=args.model_path, embedding_model=args.embedding_model)
