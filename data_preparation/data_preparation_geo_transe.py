from sklearn.model_selection import train_test_split
import pandas as pd

edge_data = pd.read_csv("data/dbpedia/triple_data.csv")
edge_label = edge_data.pop("predicate")
train_data, test_data, train_label, test_label = train_test_split(edge_data, edge_label, test_size=0.1, random_state=42)
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=1/9, random_state=42)

train_frame = pd.concat([train_data, train_label], axis=1)
train_frame = train_frame[["subject", "predicate", "object"]]
train_frame.to_csv("geo_transe/datasets/dbpedia/input/train.txt", sep="\t", index=False)

val_frame = pd.concat([val_data, val_label], axis=1)
val_frame = val_frame[["subject", "predicate", "object"]]
val_frame.to_csv("geo_transe/datasets/dbpedia/input/valid.txt", sep="\t", index=False)

test_frame = pd.concat([test_data, test_label], axis=1)
test_frame = test_frame[["subject", "predicate", "object"]]
test_frame.to_csv("geo_transe/datasets/dbpedia/input/test.txt", sep="\t", index=False)
