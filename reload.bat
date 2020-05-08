call full_reload_private_prefix.bat %1 %2
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

python manage.py edit_database_filenamehelper  --mode add_category
python manage.py edit_database_filenamehelper  --mode add_charger_drive_profile
python manage.py edit_database_filenamehelper  --mode add_experiment_type
python manage.py edit_database_filenamehelper --mode display --model Category
python manage.py edit_database_filenamehelper --mode display --model ChargerDriveProfile
python manage.py edit_database_filenamehelper --mode display --model ExperimentType


python manage.py edit_database_filenamehelper --mode just_add_files --data_dir=%3
python manage.py edit_database_filenamehelper --mode just_parse_database_files
python manage.py import_and_process_raw_neware --DEBUG