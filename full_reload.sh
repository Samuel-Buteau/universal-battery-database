rm db.sqlite3
rm -f neware_parser/migrations/*
touch neware_parser/migrations/__init__.py
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py import_raw_data --path_to_file=NEWARE_Degradation_Analysis
python3 manage.py process_raw_data