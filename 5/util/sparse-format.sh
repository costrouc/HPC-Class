filename=$1
format=$2

if [ "$format" == "CRS" ]
then
    echo "Converting to CRS format"
    head -1 data/$filename > data/$filename.crs | tail -n +2 data/$filename | sort -n -k 1,1 -k 2,2 >> data/$filename.crs
elif [ "$format" == "CCS" ]
then
    echo "Converting to CCS format"
    head -1 data/$filename > data/$filename.ccs | tail -n +2 data/$filename | sort -n -k 2,2 -k 1,1 >> data/$filename.ccs
fi
