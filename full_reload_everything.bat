del db.sqlite3
del neware_parser\migrations\*
copy NUL neware_parser\migrations\__init__.py

del FileNameHelper\migrations\*
copy NUL FileNameHelper\migrations\__init__.py
python manage.py makemigrations FileNameHelper
python manage.py makemigrations
python manage.py migrate


python manage.py edit_database_filenamehelper  --mode add_category
python manage.py edit_database_filenamehelper  --mode add_charger_drive_profile
python manage.py edit_database_filenamehelper  --mode add_experiment_type
python manage.py edit_database_filenamehelper --mode display --model Category
python manage.py edit_database_filenamehelper --mode display --model ChargerDriveProfile
python manage.py edit_database_filenamehelper --mode display --model ExperimentType



subst X: C:\Users\Samuel\Documents\LabData\Cache

python manage.py edit_database_filenamehelper --mode just_add_files ^
 --data_dir=C:\Users\Samuel\Documents\LabData\srv\samba\share\DATA

python manage.py edit_database_filenamehelper --mode just_parse_database_files

python manage.py import_and_process_raw_neware --path_to_filter=degradation_meta.file --DEBUG

