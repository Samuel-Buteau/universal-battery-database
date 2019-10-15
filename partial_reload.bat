
python manage.py edit_database
subst X: C:\Users\Samuel\Documents\LabData\Cache
subst A: C:\Users\Samuel\Documents\LabData\srv\samba\share

mkdir  X:\NEWARE_Logs\

python manage.py import_raw_data  ^
 --path_to_filter=degradation_meta.file ^
 --path_to_file=A:\DATA\CYCLING\NEWARE\ ^
 --path_to_log=X:\NEWARE_Logs\log_import ^
 --DEBUG



REM python manage.py process_raw_data --DEBUG
