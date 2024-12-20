import pandas as pd
from rdflib import Literal
from rdflib import Graph as rdfGraph
from rdflib.util import guess_format
from itertools import chain
from shapely import Point
from tqdm import tqdm


def combine_dbpedia_triples(path_list: list) -> pd.DataFrame:
    triple_list = []
    for file_path in tqdm(path_list):
        rdf_file_format = guess_format(file_path)
        if rdf_file_format is not None:
            dbpedia_kg = rdfGraph()
            dbpedia_kg.parse(file_path)
            if file_path != "data/dbpedia/geo_coordinates_mappingbased_en.ttl":
                edge_list = [{"subject": subject, "predicate": predicate, "object": obj} for subject, predicate, obj in tqdm(dbpedia_kg) if not isinstance(obj, Literal)]
            else:
                edge_list = [{"subject": subject, "predicate": predicate, "object": obj} for subject, predicate, obj in tqdm(dbpedia_kg)]
            triple_list.append(edge_list)
            dbpedia_kg.close()
        else:
            raise TypeError("Wrong file format submitted. Please check the following web page for supported file types: https://rdflib.readthedocs.io/en/stable/apidocs/rdflib.html#rdflib.util.guess_format")
    triple_list = chain.from_iterable(triple_list)
    edge_df = pd.DataFrame(triple_list, columns=['subject', 'predicate', 'object'])
    return edge_df

if __name__ == "__main__":
    kg_file_list = ["data/dbpedia/article_categories_en.ttl", "data/dbpedia/instance_types_en.ttl", "data/dbpedia/instance_types_transitive_en.ttl",
                    "data/dbpedia/mappingbased_objects_en.ttl", "data/dbpedia/skos_categories_en.ttl", "data/dbpedia/geo_coordinates_mappingbased_en.ttl"]
    dbpedia_edge_df = combine_dbpedia_triples(kg_file_list)
    # # geographic data preparation
    dbpedia_edge_df.to_csv("data/dbpedia/triple_data.csv", index=False)
    dbpedia_edge_df = pd.read_csv("data/dbpedia/triple_data.csv")
    
    dbpedia_edge_df["predicate"] = dbpedia_edge_df["predicate"].astype("str")
    geo_coordinates = dbpedia_edge_df[dbpedia_edge_df["predicate"] == 'http://www.georss.org/georss/point']
    geo_coordinates[["latitude", "longitude"]] = geo_coordinates["object"].str.split(" ", n=1, expand=True)
    geo_coordinates["geometry"] = geo_coordinates.apply(lambda row: Point(row.longitude, row.latitude).wkt, axis=1)
    geo_coordinates = geo_coordinates[["subject", "predicate", "geometry"]]
    geo_coordinates = geo_coordinates.rename({"geometry": "object"}, axis=1)
    drop_index = geo_coordinates.index
    dbpedia_edge_df = dbpedia_edge_df.drop(drop_index, axis=0)
    dbpedia_edge_df = pd.concat([dbpedia_edge_df, geo_coordinates], axis=0)
    dbpedia_edge_df = dbpedia_edge_df.sort_index()
    dbpedia_edge_df.to_csv("data/dbpedia/triple_data.csv", index=False)