# how to set things up.

TODO: somebody with more linux/mac os x knowledge should fill this part for those systems.

## Windows 10
install dependencies with 
> pip -r requirements.txt

In order to allow background tasks, 
which are super useful in this case,
run in a separate terminal:
> python manage.py process_tasks

This will process the tasks as they are defined.


Then, the more tricky part is to install postgresql and configure it. 

- make sure installation includes the PostgreSQL Unicode ODBC driver 
(after the PostgreSQL installation, there is a separate process where you can choose a driver. I selected ODBC 64-bit)

- make sure you create a user with a password you remember.

- add the bin path of the install to the Path variable.

- run the following:
> psql -U postgres

(Then, you enter the password that you hopefully still remember!!)
> CREATE DATABASE myproject;

> CREATE USER myuser WITH PASSWORD ‘mypassword’;

> GRANT ALL PRIVILEGES ON DATABASE myproject TO myuser;


- add a file called config.ini in the root directory, with the following content (feel free to modify):
>[DEFAULT]

>Database = myproject

>User = myuser

>Password = mypassword

>Host = localhost

>Port = 5432


This is for security purposes.

TODO(samuel): before releasing the database itself, also make sure the sensitive contents are removed. 


# How to implement new file formats
The code has a simple bottleneck where text files are imputted and a python data gets output. this function is called read_neware.
The output of the function should be the same for neware inputs, maccor inputs, moli inputs, etc...

The way I imagine this is a different function for each cycler vendor (neware, maccor, moli, etc...), perhaps read_neware, read_maccor, ...

And then, once the file has been given the metadata 'neware', the data import routine will call read_neware.

The output format is an ordered dict of ordered dicts of tuples with the first element of the tuple being something like 'CC_Chg' for constant current charge step, 
and the second element of the tuple is a list of lists containing voltage current capacity time data.

for instance, if data is the output, data[30][62]  would be a tuple corresponding to the step 62, cycle 30.
data[30][62][0] might be 'CC_DChg' and data[30][62][1] might be a list of lists.
data[30][62][1][3,:] would be a list like [voltage1, capacity1, current1, datetime1, accuracy_in_seconds]
here accuracy_in_seconds is a boolean variable saying whether datetime1
 should be trusted down to the seconds or down to the minutes.
 
 Note(Samuel): by the way I know this is a bit ducktapy, please feel free to use pandas or whatever you kids use these days ;P
 
 
 
 
 
# TODO(Samuel):
- various fields are always "private" in the sense that they should never be shared in the global space.
- various fields are sometimes "private" in the sense that they may or may not be shared with the global space.
- various fields are never "private".

- electrolyte and dry cell data have to be:
 - defined.
 - linked.
 - searched.
 - visualized.
 
- DOD, Rate, Temperature, 
    voltage hold (time or current), 
    open circuit, 
    CCCV, rest=open circuit, information
- factor into cyclegroups. 