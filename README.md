# how to set things up.
## without the database
1. you must install requirements with pip install -r requirements_nosql.txt
2. you must download a dataset file and put it in the appropriate folder
3. you must create a file called config.ini in neware_parser/, and put the following content within:

>[DEFAULT]

> Database = d

>User = u

>Password = p

>Host = localhost

>Port = 0000

>Backend = sqlite3

>SecretKey = verysecretkeyhaha






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
DONE - remove all the fields that I don't currently care about.
DONE - good name generation
DONE - good uniqueness check.(library)
DONE - good uniqueness check (view)
DONE - test uniqueness check
DONE - streamline the various definitions into much simpler and unique flows.


- create a separate page for:
    - electrolyte definition
    - electrode definition
    - dry cell definition
    - wet cell definition
- create SOP for entering info (especially incomplete info)
- allow modifications
- allow unknown values
- machine learning postprocess:
DONE    - make the processing specific to machine learning on-demand.
DONE    - bake a numpy structured array instead of having to use the database all the time.
    - add key variables to the numpy structured array (dod, temperature, charge/discharge CC curves)

- For temperature, first we shall target only cases where the temperature is constant across cycling.
- In general, compute a handcrafted embedding of the protocol which is the average per cycle of something.
    - for example, the average temperature, 
    - the average DOD (up, down), 
    - the average rate of charge, rate of discharge.


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

- In general, a cycle is merely a sequence of steps. 
- There is no completely general mechanism to summarize arbitrary 
sequences of steps while also being convenient for modelling.

So we simplify and we have a partial mapping from steps to cycle summary.
This mapping can evolve over time. First we must represent with a fixed lenght
 vector the sequence of steps protocol. Second, we must represent 
 the measured data in a unified format. At first, we have only CC Voltage curves (cap vs voltage).
 We can also represent CV curves as a capacity vs rate curve. 
 OCV can be represented as time vs voltage. 
 But without use cases, it is hard to know what is best. Right now, not all of this is supported.
 
 
# A Guide to entering the metadata the proper way
First, there needs to be some concept of unknown value, since the info can be missing sometimes.
TODO: list the ways to handle missing value in the definition page

TODO: when validating, null means unapplicable or unknown. there needs to be a flag to distinguish between the two.
for ratios, missing always means unknown

In terms of modelling, any value which can be Unknown should be given a latent variable per cell, electrolyte, ...
and the general way to handle this is indicator*known + (1-indicator)*latent where indicator is 0 in case of unknown and 1 in case of known and where latent is trainable.
This might be slightly inefficient, but at most 2x, and having this systematic mechanism will avoid headaches later.

As for how to enter the info the right way, below we list some examples, some general rules of thumb, and generally try to be consistent.

# Stochiometry
- It is reccomended to always use whole numbers. For instance, instead of 0.33, 0.33, 0.33, simply use 1, 1, 1
  If there are some very specific ratios that are too inexact to rationalize, you can try to have sub whole numbers.

