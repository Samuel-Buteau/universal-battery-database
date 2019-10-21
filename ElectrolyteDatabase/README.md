# Dealing with common bugs
##(1146, "Table 'filename_data.FileNameHelper_rptexperiment' doesn't exist")
Requires migration; run the following commands:
1. python manage.py makemigrations ElectrolyteDatabase
2. python manage.py migrate ElectrolyteDatabase --database=electrolytedatabase_db

##Reverse for 'experiment_overview' not found. 'experiment_overview' is not a valid view function or pattern name.
The line was:
  return HttpResponseRedirect(reverse('FileNameHelper:experiment_overview')) 
The problem is in urls.py, there are no urls with name=experiment_overview

The line was:
<li><a href="{% url 'FileNameHelper:novonix_experiment_overview' %}">Novonix Experiment Overview</a></li>
The problem is in urls.py, there are no urls with name=novonix_experiment_overview

##name 'NewareExperimentFormSet' is not defined
The line was:
formset = NewareExperimentFormSet() 
The problem is in views.py, probably a copy paste error. Most common is missing import or typo in vareiable name

##(1364, "Field 'upper_cutoff_voltage' doesn't have a default value")
This happened because I tried to enter something into the database and one of the required fields is not present.
Migration is required, refer to first bug on this list. 

##(1054, "Unknown column 'FileNameHelper_rptexperiment.storage_potential' in 'field list'")
The model defines a variable that is not in the database yet, need to do a migration (refer to first bug on this list.)

##How to wipe database
Delete all files in 'migrations' except __init__
Delete ...db.sqlite in root_production

## After Shutting Down your computer
If you shut down your computer, run 'vagrant reload' in your Mac and then enter the Virtual Env. 
