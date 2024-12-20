# GeoRDF2Vec
This GitHub repository provides the code base for the paper **GeoRDF2vec – Learning Location-Aware Entity
Representations in Knowledge Graphs**. In the following, we will explain the outline of our coding base together with the necessary parameters to run the experiments.

## Tracking of experiment runs
In order to automate the experimentation deployment as well as the logging of our experiment results we use MLflow as a tracking platform. It uses a local host together with a SQLite database where the runs are stored. The models itself are stored in a folder on a file storage called artifact. Each experiment receives a unique identifier for further processing.

## Download of datasets
For the download of the datasets the following sections elaborate on the download for both datasets with KGbench and DBPedia
### DBPedia download
In order to generate the relevant subset of DBPedia, please run first the following [shell script](data_preparation/dbpedia_download.sh) to download the data. After this, please run the following script to prepare the triple dataset for the runs
### KGBench download
The instructions for generating the KGbench repository our outlined on their [kgbench website](http://kgbench.info) and [GitHub repository](https://github.com/pbloem/kgbench-loader). For the dmg777k move the generated datasets to [data/kgbench_dmg777k](data/kgbench_dmg777k).
## GeoRDF2Vec implementation
