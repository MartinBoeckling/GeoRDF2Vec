from pathlib import Path
from dmg_hpt import modelPrediction
from tqdm import tqdm

word2vec_models = Path("model/kgbench_dmg777k").glob('*')
word2vec_models = sorted(list(word2vec_models))
# print(word2vec_models)
for model_path in tqdm(word2vec_models):
    model_path = str(model_path)
    modelPrediction("data/kgbench_dmg777k", False, str(model_path))
