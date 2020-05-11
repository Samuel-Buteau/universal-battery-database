(
echo drop database $1;
echo create database $1 with owner $2;
echo \q
) | psql -U postgres
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

python3 manage.py edit_database_filenamehelper  --mode add_category
python3 manage.py edit_database_filenamehelper  --mode add_charger_drive_profile
python3 manage.py edit_database_filenamehelper  --mode add_experiment_type
python3 manage.py edit_database_filenamehelper --mode display --model Category
python3 manage.py edit_database_filenamehelper --mode display --model ChargerDriveProfile
python3 manage.py edit_database_filenamehelper --mode display --model ExperimentType


python3 manage.py edit_database_filenamehelper --mode just_add_files --data_dir=$3
python3 manage.py edit_database_filenamehelper --mode just_parse_database_files
python3 manage.py import_and_process_raw_neware --DEBUG