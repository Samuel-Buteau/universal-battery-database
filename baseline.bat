subst X: C:\Users\Samuel\Documents\LabData\Cache
subst A: C:\Users\Samuel\Documents\LabData\srv\samba\share

mkdir  X:\NEWARE_Logs\
mkdir  X:\NEWARE_Cached\
mkdir  X:\NEWARE_Processed_Step\
mkdir  X:\NEWARE_Processed_Cycle\
mkdir  X:\NEWARE_Processed_Rate\
mkdir  X:\NEWARE_Processed_Separated\
mkdir  X:\NEWARE_Processed_FirstRobyPattern\
mkdir  X:\NEWARE_Processed_SecondRobyPattern\
mkdir  X:\NEWARE_Processed_VvsQ\
mkdir  X:\NEWARE_Processed_RateVvsQ\
mkdir  X:\NEWARE_Degradation_Analysis\

python smoothing.py ^
--path_to_degradation_analysis=X:\NEWARE_Degradation_Analysis\ ^
--common_path=''

