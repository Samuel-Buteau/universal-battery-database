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


