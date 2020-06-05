python3 manage.py edit_database_filename_database --mode just_add_files --data_dir=$3
python3 manage.py edit_database_filename_database --mode just_parse_database_files
python3 manage.py import_and_process_raw_neware --DEBUG