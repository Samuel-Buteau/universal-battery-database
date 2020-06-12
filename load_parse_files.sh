python3 manage.py edit_database_filename_database --mode just_add_files --data_dir=$1
python3 manage.py edit_database_filename_database --mode just_parse_database_files
python3 manage.py import_and_process_raw_neware --NO_DEBUG --max_filesize=1000000000 --log_dir=logging
