# import packages
import pandas as pd
import geopandas as gpd
from shapely import from_wkt

def kgbench_dmg_prep(kg_version: str) -> None:
    """_summary_

    Args:
        kg_version (str): _description_
    """
    triples_data = pd.read_csv(f"data/kgbench_{kg_version}/triples.int.csv", header=None, names=["subject", "predicate", "object"])
    node_labels = pd.read_csv(f"data/kgbench_{kg_version}/nodes.int.csv")
    relation_label = pd.read_csv(f"data/kgbench_{kg_version}/relations.int.csv")
    triples_data = pd.merge(triples_data, node_labels, how="inner", left_on="subject", right_on="index")
    triples_data = triples_data.drop(labels=["subject", "annotation"], axis=1)
    triples_data = triples_data.rename({"label": "subject", "index": "subject_index"}, axis=1)
    triples_data = pd.merge(triples_data, node_labels, how="inner", left_on="object", right_on="index")
    triples_data = triples_data.drop(labels=["object", "annotation"], axis=1)
    triples_data = triples_data.rename({"label": "object", "index": "object_index"}, axis=1)
    triples_data = pd.merge(triples_data, relation_label, how="inner", left_on="predicate", right_on="index")
    triples_data = triples_data.drop(labels=["predicate"], axis=1)
    triples_data = triples_data.rename({"label": "predicate", "index": "predicate_index"}, axis=1)
    triples_data = triples_data[["subject", "predicate", "object"]]
    rd_geometries = triples_data[triples_data["predicate"] == "http://data.pdok.nl/def/pdok#asWKT-RD"]
    rd_geometries.loc[:,"object"] = from_wkt(rd_geometries["object"], on_invalid='ignore')
    gpd_rd_geometries = gpd.GeoDataFrame(rd_geometries, geometry="object")
    gpd_rd_geometries = gpd_rd_geometries.set_crs(epsg=28992)
    gpd_rd_geometries_projected = gpd_rd_geometries.to_crs(epsg=4326)
    transformed_geometries = gpd_rd_geometries_projected.to_wkt()
    transformed_geometries_index = transformed_geometries.index
    triples_data = triples_data.drop(transformed_geometries_index, axis=0)
    triples_data = pd.concat([triples_data, transformed_geometries], axis=0)
    triples_data = triples_data.dropna()
    triples_data.to_csv(f"data/kgbench_{kg_version}/full_triple_set.csv", index=False)


if __name__ == "__main__":
    kg_paths = ["dmg777k"]
    for kg_path in kg_paths:
        kgbench_dmg_prep(kg_version=kg_path)