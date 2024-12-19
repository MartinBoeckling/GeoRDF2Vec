import multiprocessing as mp
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from collections import defaultdict
from itertools import groupby, chain, product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Process
from shapely import from_wkt, GeometryCollection, Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, Geometry
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from igraph import Graph
from pathlib import Path
from rdflib import Graph as rdfGraph
from rdflib.util import guess_format
from gensim.models.word2vec import Word2Vec as W2V
import pickle

class GeoRDF2Vec:
    def __init__(self, cpu_count, file_path, flood_direction, chunksize, max_walks, distance, walk_extraction, geo_implementation, spatial_weighting_strategy) -> None:
        self.cpu_count = cpu_count
        self.file_path = Path(file_path)
        self.model_base_path = str(self.file_path.parent).split("/")[-1]
        self.walk_extraction = walk_extraction
        self.save_dir = Path(f'model/{self.model_base_path}_normalized/{geo_implementation}_{walk_extraction}_{flood_direction}_{distance}_{max_walks}_{spatial_weighting_strategy}')
        self.chunksize = chunksize
        self.flood_direction = flood_direction
        self.geometry_cache = {}
        self.max_walks = max_walks
        self.distance = distance
        self.geo_implementation = geo_implementation
        self.spatial_weighting_strategy = spatial_weighting_strategy
        self.kg_training()

    def extract_geometry(self, node_label: str) -> tuple:
        """Method in order to extract geometries from Node labels which do not contain single geometries.
        It uses a regex pattern to extract from node labels the respective geometry. It assumes a correct WKT 
        storage of the geometry as a node label

        Args:
            node_label (str): label of node of Knowledge Graph

        Returns:
            tuple: Tuple containing a boolean value and string value. 
                If the regex pattern does not discover a WKT geometry, we assign to first position false and at second position empty string. 
                If the regex pattern discovers a WKT geometry, we assign to first position true and at second position regex extracted string.
        """
        # regex for overall extraction of geometries if string is not a real WKT geometry
        node_label = str(node_label)
        extracted_geometry = re.search(r'(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)(\s?\(.+)(\))', node_label)
        # If overall extracted geometry is not an empty list, set to true. Otherwise, the method returns false
        if extracted_geometry:
            return (True, extracted_geometry[0])
        else:
            return (False, "")

    def read_kg_file(self, file_path: str) -> None:
        """Warpper method to read different KG files in. If the edge dataframe uses a rdflib supported file type,
        rdflib is used. In case the KG is provided in a DeltaTable format, those readers it is used to read the file in.

        Args:
            file_path (str): File path for knowledge graph file

        Returns:
            Graph: Return tuple structure containing triple structure using the 
        """
        rdf_file_format = guess_format(file_path)
        if rdf_file_format is not None:
            kg = rdfGraph()
            kg.parse(file_path)
            kg.close()
            # prepare knowledge graph
            edge_list = [triple for triple in kg]
            edge_df = pd.DataFrame(edge_list, columns=['subject', 'predicate', 'object'])
        else:
            edge_df = pd.read_csv(file_path)
        edge_df = edge_df[['subject', 'object', 'predicate']]
        edge_df = edge_df.to_records(index=False)
        kg_graph = Graph().TupleList(edges=edge_df, directed=True, edge_attrs=["predicate"])
        return kg_graph

    def stream_neighbors(self, graph: Graph, geo_nodes: list, order: int):
        for geo_node in geo_nodes:
            yield geo_node, graph.neighborhood(vertices=[geo_node], order=order)


    def process_chunk(self, geo_nodes_batch, graph, geometry_cache, visited_node_graph, order, results):
        """
        Process a chunk of geo_nodes to search for neighbors.
        """
        local_geo_vertex_dict = defaultdict(list)
        local_current_neighbors = set()

        # Process multiple geo_nodes at once
        batch_neighbors = graph.neighborhood(vertices=geo_nodes_batch, order=order)
        for geo_node, geo_neighbors in zip(geo_nodes_batch, batch_neighbors):
            _, geometry = geometry_cache[geo_node]
            geometry = from_wkt(geometry)
            for geo_neighbor in geo_neighbors:
                if geo_neighbor not in visited_node_graph:
                    bool_geo_neighbor, _ = geometry_cache[geo_neighbor]
                    if not bool_geo_neighbor:
                        local_geo_vertex_dict[geo_neighbor].append(geometry)
                    elif geo_neighbor == geo_node:
                        local_geo_vertex_dict[geo_neighbor].append(geometry)
                    local_current_neighbors.add(geo_neighbor)

        # Append results to shared list
        results.append((local_geo_vertex_dict, local_current_neighbors))
    

    def neighborhood_flood(self, geo_nodes) -> defaultdict:
        """Flood the neighborhood starting from geographic nodes. Whenever a vertex already has been visited in the past, the geometry is not assigned.
        In case the visited vertex is not a geographic vertex and has not been visited before, it gets a geometry assigned. In the end, all nodes have a 
        geography assigned to itself.

        Args:
            geo_nodes (list): list of indices which contain geographic geometries

        Returns:
            defaultdict: return of nodes with assigned geometries as keys and assigned geometries as elements in the list
        """
        # initialize variables to method variables
        graph = self.graph
        geometry_cache = self.geometry_cache
        graph_flood = True
        order = 1
        geo_vertex_dict = defaultdict(list)

        # Use multiprocessing Manager for shared state
        manager = Manager()
        visited_node_graph = set()  # Use a regular Python set for visited nodes
        previous_length_set = 0
        num_processors = self.cpu_count

        while graph_flood:
            current_neighbors = set()  # Use a local set for this order

            # Dynamically calculate batch size to balance memory usage
            batch_size = max(1, math.ceil(len(geo_nodes) / (num_processors * 4)))  # 4x tasks per processor
            batches = [geo_nodes[i:i + batch_size] for i in range(0, len(geo_nodes), batch_size)]

            # Use multiprocessing to parallelize batch processing
            processes = []
            results = manager.list()
            for batch in batches:
                p = Process(
                    target=self.process_chunk,
                    args=(batch, graph, geometry_cache, visited_node_graph, order, results)
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            # Aggregate results from all processes
            for local_geo_vertex_dict, local_current_neighbors in results:
                for key, value in local_geo_vertex_dict.items():
                    geo_vertex_dict[key].extend(value)
                current_neighbors.update(local_current_neighbors)

            # Update global visited nodes
            visited_node_graph.update(current_neighbors)

            # Check for convergence
            length_set = len(visited_node_graph)
            if previous_length_set == length_set:
                break
            previous_length_set = length_set
            order += 1

        return geo_vertex_dict
    
    def determine_optimal_chunksize(self, length_iterable: int) -> int:
        """Method to determine optimal chunksize for parallelism of unordered method

        Args:
            length_iterable (int): Size of iterable

        Returns:
            int: determined chunksize
        """
        chunksize, extra = divmod(length_iterable, self.cpu_count * 4)
        if extra:
            chunksize += 1
        return chunksize

    def predicate_generation(self, path_list: str) -> list:
        """Generates a sequence of predicates for a given path.

        Args:
            path_list (str): The path for which to generate predicates.

        Returns:
            list: A list of predicates for the given path.
        """
        # assign class graph to graph variable
        graph = self.graph
        # extract predicate of edge given edge id stored in numpy
        pred_values = [e.attributes()['predicate'] for e in graph.es(path_list)]
        # extract node sequences that are part of the edge path and flatten numpy array
        node_sequence = np.array([graph.vs().select(e.tuple).get_attribute_values(
            'name') for e in graph.es(path_list)]).flatten()
        # delete consecutive character values in numpy array based from prior matrix
        node_sequence = np.array([key for key, _group in groupby(node_sequence)]).tolist()
        # combine predicate values and node sequences to one single array
        if node_sequence:
            path_sequence = []
            for index, value in enumerate(node_sequence):
                node_label = value
                edge_label = pred_values[index]
                path_sequence.append(node_label)
                path_sequence.append(edge_label)
                if index >= len(pred_values) -1:
                    last_value = node_sequence[-1]
                    path_sequence.append(last_value)
                    break
        else:
            path_sequence = []
        # return path sequence numpy array
        return path_sequence


    def random_walk_iteration(self, start_vertex: int) -> list:
        """_summary_

        Args:
            start_vertex (int): Start vertex starting the random walk from

        Returns:
            list: Walk list with length defined as number of walks
        """
        walk_list = []
        for _ in range(self.max_walks):
            walk_edges = self.graph.random_walk(start=start_vertex, steps=self.distance, return_type="edges", weights=self.graph.es["weight"])
            walk_predicates = self.predicate_generation(walk_edges)
            walk_list.append(walk_predicates)
        return walk_list

    def geom_preparation(self, geometry_tuple: tuple) -> tuple:
        """_summary_

        Args:
            dict_key (int): _description_
            geom_dict (defaultdict): _description_

        Returns:
            list: _description_
        """
        vertex_id, geometry_list = geometry_tuple
        if len(geometry_list) > 1:
            centroid_geometry = [GeometryCollection(geoms=geometry_list).centroid]
        else:
            centroid_geometry = geometry_list
        return (vertex_id, centroid_geometry)

    def extract_point_coordinates(self, geom: Geometry) -> list:
        """For non Point geometries we use the centroid of a geometry in order to calculate the geodesic distance between geometries.

        Args:
            geom (Geometry): Input geometry for calculation of centroid. In case of a point geometry, no point geometry is calculated

        Raises:
            ValueError: In case the geometry is not supported a Value Error is raised

        Returns:
            list: List of Longitude, Latitude tuple pair
        """
        if geom.is_empty:
            return []
        
        if isinstance(geom, Point):
            return [(geom.x, geom.y)]
        elif isinstance(geom, (LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
            geom_centroid = geom.centroid
            return [(geom_centroid.x, geom_centroid.y)]
        else:
            raise ValueError("Given input represents an unsupported geometry type")

    def spatial_distance_calculation(self, geom1: list, geom2: list) -> float:
        """Method calculating the spatial weighting of 2 geometries. In order to calculate the geodesic distance,
        non point geometries are transformed by being represented of their 

        Args:
            geom1 (list): List containing first geometry for distance calculation
            geom2 (list): List containing second geometry for distance calculation

        Returns:
            float: Spatial weight based on great circular distance
        """
        # check if both input parameters contains non empty list
        # exctract both geometries from lists
        # extracts centroid to calculate the spherical distance
        if geom1 and geom2:
            geom1 = geom1[0]
            geom2 = geom2[0]
            coords1 = self.extract_point_coordinates(geom1)
            coords2 = self.extract_point_coordinates(geom2)
            # calculate geodesic distance between centroids of geom1 and geom2 in kilometers
            try:
                distance = geodesic(coords1, coords2).kilometers
            except:
                distance = 0
        else:
            distance = 0
        return distance


    def spatial_weighting_calculation(self, geom1: list, geom2: list) -> float:
        """Method calculating the spatial weighting of 2 geometries. In order to calculate the geodesic distance,
        non point geometries are transformed by being represented of their 

        Args:
            geom1 (list): List containing first geometry for distance calculation
            geom2 (list): List containing second geometry for distance calculation

        Returns:
            float: Spatial weight based on great circular distance
        """
        # check if both input parameters contains non empty list
        if geom1 and geom2:
            # exctract both geometries from lists
            geom1 = geom1[0]
            geom2 = geom2[0]
            # extracts centroid to calculate the spherical distance
            coords1 = self.extract_point_coordinates(geom1)
            coords2 = self.extract_point_coordinates(geom2)
            # calculate geodesic distance between centroids of geom1 and geom2 in kilometers
            distance = geodesic(coords1, coords2).kilometers
            # calculate spatial weighting by using exp function with negative distance 
            spatial_weight = math.exp(-distance)
            return spatial_weight
        else:
            return 1.0

    def kg_training(self) -> None:
        """_summary_
        """
        self.graph = self.read_kg_file(file_path=str(self.file_path))
        vertex_index_list = [vertex.index for vertex in self.graph.vs]
        if self.geo_implementation:  
            self.geometry_cache = {vertex.index: self.extract_geometry(vertex['name']) for vertex in self.graph.vs}
            geo_nodes = [key for key, value in self.geometry_cache.items() if value[0]]
            if len(geo_nodes) == 0:
                raise AttributeError("No geographic entity found, therefore only normal RDF2Vec needed")

            geo_vertex_dict = self.neighborhood_flood(geo_nodes)
            geo_tuple_list = tuple(geo_vertex_dict.items())
            chunksize = self.determine_optimal_chunksize(len(geo_tuple_list))
            with mp.Pool(self.cpu_count) as pool:
                geo_vertex_tuple = tuple(tqdm(pool.imap_unordered(self.geom_preparation, geo_tuple_list, chunksize=chunksize), desc="Geom centroid", total=len(geo_tuple_list)))

            geo_vertex_list = defaultdict(list, geo_vertex_tuple)
            spatial_weighting_strategy = self.spatial_weighting_strategy
            if spatial_weighting_strategy == "naive":
                # create weights for graph
                for edge in tqdm(self.graph.es, desc="Graph weighting"):
                    source_node, target_node = edge.source, edge.target
                    source_geom, target_geom = geo_vertex_list[source_node], geo_vertex_list[target_node]
                    weighting = self.spatial_weighting_calculation(source_geom, target_geom)
                    edge["weight"] = weighting
            elif spatial_weighting_strategy == "min_max_normalization":
                for edge in tqdm(self.graph.es, desc="Graph edge distance"):
                    source_node, target_node = edge.source, edge.target
                    source_geom, target_geom = geo_vertex_list[source_node], geo_vertex_list[target_node]
                    distance = self.spatial_distance_calculation(source_geom, target_geom)
                    edge["distance"] = distance

                edge_distance_df = self.graph.get_edge_dataframe()
                for graph_node in tqdm(self.graph.vs, desc="Graph node weighting normalization"):
                    node_id = graph_node.index
                    node_neighbors = self.graph.neighbors(node_id, mode="all")
                    if node_neighbors:
                        neighbor_edge_data = edge_distance_df[(((edge_distance_df["source"] == node_id) & (edge_distance_df["target"].isin(node_neighbors))) | ((edge_distance_df["source"].isin(node_neighbors)) & (edge_distance_df["target"] == node_id)))]
                        neighbor_distances = neighbor_edge_data["distance"].tolist()
                        neighbor_distances = np.array(neighbor_distances).reshape(-1,1)
                        min_max_scaler = MinMaxScaler()
                        normalized_distances = min_max_scaler.fit_transform(neighbor_distances)
                        normalized_distances = normalized_distances.flatten().tolist()
                        weights = [math.exp(-distance) for distance in normalized_distances]
                        edge_ids = neighbor_edge_data.index.tolist()
                        edge_spatial_weights = zip(edge_ids, weights)
                        for edge_id, weight in edge_spatial_weights:
                            self.graph.es[edge_id]["weight"] = weight
        else:
            self.graph.es["weight"] = 1
        # write weighted graph out
        self.graph.write_pickle("dbpedia_weighted_graph.pkl")
        chunksize = self.determine_optimal_chunksize(len(vertex_index_list))
        
        # walk_distance_range = list(range(2, 11))
        # number_walk_range = list(range(100, 600, 100))
        walk_distance_range = [8]
        number_walk_range = [100]
        combined_parameters = list(product(walk_distance_range, number_walk_range))
        for combined_parameter in combined_parameters:
            self.distance = combined_parameter[0]
            self.max_walks = combined_parameter[1]
            self.save_dir = Path(f'model/{self.model_base_path}/{self.geo_implementation}_{self.walk_extraction}_{self.flood_direction}_{self.distance}_{self.max_walks}_{self.spatial_weighting_strategy}')
            self.save_dir.mkdir(parents=True, exist_ok=True)
            chunksize = self.determine_optimal_chunksize(len(vertex_index_list))
            with mp.Pool(self.cpu_count) as pool:
                walk_predicate_list = list(tqdm(pool.imap_unordered(self.random_walk_iteration, vertex_index_list, chunksize=chunksize), desc="Walk iteration", total=len(vertex_index_list)))
            corpus = chain.from_iterable(walk_predicate_list)
            model = W2V(min_count=0, workers=self.cpu_count, seed=15, vector_size=200)
            self.model = model
            model.build_vocab(corpus)
            model.train(corpus, total_examples=model.corpus_count, epochs=10)
            model.save(f"{str(self.save_dir)}/word2vec.model")

if __name__ == "__main__":
    geoRDF2Vec_instance = GeoRDF2Vec(120, "data/kgbench_dmg777k/full_triple_set.csv", "all", 50, 2, 10, "random", False, "min_max_normalization")
