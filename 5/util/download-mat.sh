if [ ! -d "data/" ];
then
    mkdir data/
fi

wget -P data/. http://web.eecs.utk.edu/~dongarra/WEB-PAGES/SPRING-2009/matrix.output.gz
gunzip -c data/matrix.output.gz > data/matrix.output

python util/reorder_matrix.py

./util/sparse-format.sh matrix.output CRS
./util/sparse-format.sh matrix.output CCS
./util/sparse-format.sh matrix.reorder CRS
./util/sparse-format.sh matrix.reorder CCS

