import pickle
import pandas as pd
import numpy as np

with open("geo_transe/datasets/kgbench/result/entity_embeddings.pickle", "rb") as file:
    embedding_data = pickle.load(file)

with open("geo_transe/datasets/kgbench/output/entity2id.pickle", "rb") as file:
    entity_id = pickle.load(file)

embedding_numpy = embedding_data.cpu().detach().numpy()
embedding_df = pd.DataFrame(embedding_numpy)
embedding_df = embedding_df.add_prefix("vector_", axis=1)

id_entity = [(value, key) for key, value in entity_id.items()]
id_entity_df = pd.DataFrame(id_entity, columns=["id", "entity"])
entity_embedding_df = pd.merge(id_entity_df, embedding_df, how="inner", left_index=True, right_index=True)
nodes_df = pd.read_csv("data/kgbench_dmg777k/nodes.int.csv")
entity_embedding_df = pd.merge(nodes_df, entity_embedding_df, how="inner", left_on="label", right_on="entity")

entity_embedding_df = entity_embedding_df.drop(labels=["annotation", "label", "id", "entity"], axis=1)
entity_embedding_df.to_csv("data/kgbench_dmg777k/transgeo_embedding.csv", index=False)