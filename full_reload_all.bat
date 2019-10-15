del db.sqlite3
del neware_parser\migrations\*
copy NUL neware_parser\migrations\__init__.py

del FileNameHelper\migrations\*
copy NUL FileNameHelper\migrations\__init__.py
python manage.py makemigrations FileNameHelper
python manage.py makemigrations
python manage.py migrate

