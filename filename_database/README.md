# Dealing with common bugs
##(1146, "Table 'filename_data.FileNameHelper_rptexperiment' doesn't exist")
Requires migration; run the following commands:
1. python manage.py makemigrations FileNameHelper
2. python manage.py migrate FileNameHelper --database=filenamehelper_db

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

# To do list:
~~1. implement categories~~ 
~~1. fix 'test' in drive profile specify page. Database error~~
~~1. drive profile incompatibility page~~
~~1. Fix model issue. All experiments use Neware model and therefore Neware convention~~
~~1. Get rid of modify pages~~
~~2. allow meta data to filename without adding to database~~
~~1. Finish filename page (ie: go back buttons)~~
~~1. remove 'experiment type' field in meta data input page~~
~~1. create an entry in the database for each file in the file system and store the root and the filename~~
~~1. Fix drive profile in filename_view~~
~~2. finish parser~~
~~1. Add error page for choosing a drive profile when there isn't one~~
~~10. create help page for all parts of the website - should add entries if you add to the database~~
~~2. Run parser on database file entries~~
~~1. Boolean 'valid' object in NewareExperiment model~~
~~3. Clean up parser and remove all useless aspects~~
~~4. Fix drive profile data storage in parser and database (models)~~
~~2. Add 'charger' (ask what charger is) to Cycler in formation meta data extracto~~r
~~4. extract meta data from parser~~
~~5. allow searching of file entries based on category, exp type, etc. (ex: failure of parser, username, etc.)~~
~~6. create an entry in the database for a meta data that points to a filename entry~~
~~6. Fix student issues (if charID is not recognized, call the file invalid (and why its invalid))~~

~~1. Make string search case insensitive~~ 
  ~~2. Create non-deterministic parser - use tags~~
  
~~3. Bring search results directly onto the homepage so that users can continue to refine their search~~
~~1. Put search results in parallel column to the field in the nd parser~~
~~2. Include string search in the nd parser~~


~~3. Include experiment type in the nd parser~~
~~1. Finish user interface for nd parser~~
~~2. Finish nd logic and get it working on the site~~
~~3. Add the rest of the fields/field ranges to nd parser.~~ 
~~4. String search should be in the same page as the non-deterministic parser~~
~~5. Add date widget to date ranges~~


File name get
~~1. GET RID OF exp type and drive profile form the specify options~~
2. Give fields 'boxes'
3. Give date the date widget



1. Put box around search results on both search pages

1. Display all experiment types with the same names (ie:neware and neware FRA)
~~2. Combine the search results for the specific search page with the fields page~~
5. Learn about insensitive matches query django
1. Fix drive profile in parser
1. Remove 'No drive profile' from options on drive profile error page
3. get the valid files/ invalid (separate into different groups)
7. allow people to modify this meta data
8. use the parser to suggest some fields of the meta data of a file
9. Perfect parser -- fix issues commented at the top of the 'parsing_functions' file
