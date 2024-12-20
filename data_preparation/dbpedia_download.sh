# Download and decompress article categories
filename=("article_categories_en.ttl.bz2 instance_types_en.ttl.bz2 instance_types_transitive_en.ttl.bz2 mappingbased_objects_en.ttl.bz2 skos_categories_en.ttl.bz2 geo_coordinates_mappingbased_en.ttl.bz2")

for file in $filename; do
    # set base parameters for script
    base_url='https://downloads.dbpedia.org/2016-04/core/'
    base_path='data/dbpedia/'
    # concatenate string costants and list files
    file_url="${base_url}${file}"
    output_path="${base_path}${file}"
    # download data files und unzip files
    wget $file_url -O $output_path
    bzip2 -d $output_path
done