(
echo drop database %1;
echo create database %1 with owner %2;
echo \q
) | psql -U postgres

del cell_database\migrations\*
copy NUL cell_database\migrations\__init__.py

del cycling\migrations\*
copy NUL cycling\migrations\__init__.py

del filename_database\migrations\*
copy NUL filename_database\migrations\__init__.py

del machine_learning\migrations\*
copy NUL machine_learning\migrations\__init__.py

python manage.py makemigrations
python manage.py migrate

python manage.py edit_database_filename_database  --mode add_category
python manage.py edit_database_filename_database  --mode add_charger_drive_profile
python manage.py edit_database_filename_database  --mode add_experiment_type
python manage.py edit_database_filename_database --mode display --model Category
python manage.py edit_database_filename_database --mode display --model ChargerDriveProfile
python manage.py edit_database_filename_database --mode display --model ExperimentType


python manage.py edit_database_filename_database --mode just_add_files --data_dir=%3
python manage.py edit_database_filename_database --mode just_parse_database_files
python manage.py import_and_process_raw_neware --DEBUG