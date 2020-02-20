[ -d $1 ] && rm -r $1

mkdir $1
echo $2 > $1/notes.txt
python3 manage.py ml_smoothing --path_to_dataset compiled_datasets1 --dataset_version TESTING1 --path_to_plots=$1

