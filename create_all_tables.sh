rm db.sqlite3
rm -f neware_parser/migrations/*
touch neware_parser/migrations/__init__.py

rm -f FileNameHelper/migrations/*
touch FileNameHelper/migrations/__init__.py
python3 manage.py makemigrations FileNameHelper
python3 manage.py makemigrations
python3 manage.py migrate


python3 manage.py edit_database_filenamehelper  --mode add_category
python3 manage.py edit_database_filenamehelper  --mode add_charger_drive_profile
python3 manage.py edit_database_filenamehelper  --mode add_experiment_type
python3 manage.py edit_database_filenamehelper --mode display --model Category
python3 manage.py edit_database_filenamehelper --mode display --model ChargerDriveProfile
python3 manage.py edit_database_filenamehelper --mode display --model ExperimentType


