(
echo "DROP database $1 ;";
echo "CREATE DATABASE $1 WITH OWNER $2 ;";
echo "\q";
) | sudo -u postgres psql
rm -f cell_database/migrations/*
touch cell_database/migrations/__init__.py

rm -f cycling/migrations/*
touch cycling/migrations/__init__.py

rm -f filename_database/migrations/*
touch filename_database/migrations/__init__.py

rm -f machine_learning/migrations/*
touch machine_learning/migrations/__init__.py

python3 manage.py makemigrations
python3 manage.py migrate

python3 manage.py edit_database_filename_database  --mode add_category
python3 manage.py edit_database_filename_database  --mode add_charger_drive_profile
python3 manage.py edit_database_filename_database  --mode add_experiment_type
python3 manage.py edit_database_filename_database --mode display --model Category
python3 manage.py edit_database_filename_database --mode display --model ChargerDriveProfile
python3 manage.py edit_database_filename_database --mode display --model ExperimentType


python3 manage.py edit_database_filename_database --mode just_add_files --data_dir=$3
python3 manage.py edit_database_filename_database --mode just_parse_database_files
python3 manage.py import_and_process_raw_neware --DEBUG