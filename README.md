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
 
 
# Theoretical Overview 
 Most abstractly, we have cells and we have experimental observations, and the goal is to learn the mapping between the two.
 
 For simplicity, take the case where all experimental observations are of long term cycling. 
 
 we can imagine each distinct cell has some hidden feature vector F. Then, for every observed cycle number, we have many observed voltages for which the capacity is observed (it becomes our prediction target).
 The mapping from F, cyc, V to Cap depends on two factors:
  - the general cumulative cycling conditions, such as the average current used for charge/discharge, the average depth of charge/discharge, as well as the average temperature.
  - the specific conditions for the given cycle (the actual currents for this cycle, actual depth of discharge, actual temperature, etc.)
 
 This mapping, though a black box at first, can be broken down into more structured sub-components (we call this "mechanism-level").
 
 Also, the feature representation of a cell F itself can be broken down. First, a cell has subcomponents: Electrolyte, DryCell.
 Therefore, each component can be represented by feature vectors E, D, and a mapping from (E,D) to F may be learned.
 This level of description (level 1) forces some generalization since the number of dry cells is much smaller than the number of cells.
 
- We can break down further DryCell as a combination of Anode, Cathode, Separator, and Geometry.
- We can also break down further Electrolyte as a weighted combination of Molecules. This is a great help to generalization since 700 electrolytes can be expressed as a combination of 30 molecules, with 500 electrolytes using less than 10, and generally the combinations are sparse.
- Similarly, the Anode and Cathode has some ElectrodeGeometry, as well as a weighted combination of Materials (either ActiveMaterials or InactiveMaterials).
- The Separator has some SeparatorGeometry as well as a weighted combination of Materials.
- The ActiveMaterials have some numerical features as well as a weighted combination of atoms.
- The Molecules have a graph of Atoms. 
- the Geometry can be characterised numerically.

In this way, there is a directed acyclic graph connecting these various entities, and we can have multiple levels of descriptions.
In turn, each level of description is limited in the kinds of generalizations possible. For instance, if we stop at F, we can't know anything about what would happed to a different cell, even if the dataset was rich enough to take a good guess.
Whereas if we stop at (D,E), then we can make predictions for combinations of electrolytes with drycells that were not directly within the dataset. 
 
 
Given the generic cycling conditions (the sufficient statistics of cycling protocol SSCP) and a cycle number, we can compute a universal representation of the accumulated stress (URAS) the cell has experienced.
Furthermore, the predicted Capacity can only depend on the cycle number *through* URAS. This allows apples-to-apples comparison of different cells, as well as some generalization of the results to counter-factual scenarios where a cell would have been cycled under different conditions.
 
# TODO(Samuel):

We want two objects to never have the same name.
we want two names to never have the same object. 

if an object is a list of pairs of labels and objects or it is just a "leaf" object
so 
data Obj = Leaf x | Collection [(label, Bool, Obj)]

and we want to define a relationship of equality between objects which is the induced relationship from the equality on strings.
however, we want to do this without computing strings.
The only thing we need is the following:
- equality defined on the leaf objects.
- we need to know EQNulls = (Visible, Null) == (Invisible, Null) or not. Maybe safer to never allow this. 
- then, 
eq (Visible, x) (Visible, y) = eq x y
eq (Invisible, x) (Visible, y) = if EQNulls then ((x is Null) and (y is Null)) else False
eq (Visible, x) (Invisible, y) = if EQNulls then ((x is Null) and (y is Null)) else False
eq (Invisible, x) (Invisible, y) = True


eq (Visible, x) (Visible, y) = eq x y
eq (Visible, x) (Invisible, y) = False

eq (Invisible, x) (Invisible, y) = True
eq (Invisible, x) (Visible, y) =  False



eq {x1,x2, x3} {y1,y2, y3}= all {x1==y1, x2==y2, x3==y3} 

We also have a notion of equality which is Object equality without the visibility constraints.
i.e. 
eq2 (a, x) (b, y) = eq2 x y

Based on these two properties, we always want to maintain the propriety that the list of objects in the database is unique with respect to eq and to eq2.

## object creation
We have 3 modes of creation: 
1. create new: if object already exists, don't do anything and return existing. Otherwise, create and return, but if string already exists, set to all visible
2. override visibility: if object already exists, modify visibility flags. If new string already exists, keep old visibility flags, otherwise modify.
3. modify a specific target. First, exclude the target id from the search and do the same thing as "create new" except modify rather than create and use the target id 

We never want to create an object if there is another object satisfying object equality (eq2) in the database.
if given such an object, we return one of the equality set (eq2) and don't create anything.

From this point, assume eq2 does not hold.
We never want to create an object if there is another object satisfying string equality (eq) in the database. 
However, this needs a modification of the visibility fields, 
so we can create the object with full top level visibility and give a warning. 
If string equality is satisfied before, and this object is not object equal to anything,
 then at least one of the subobjects must be not object equal, but if that object is not object equal, 
 then it is not string equal. So visibility will distinguish them. 
 This is why you only need to set to visible at top level. 

Then, when TODO...

- The name should be unique
- allow modification of the naming fields. 
- allow general modification (define if possible with optional argument of the id)


DONE - Instead of Field, FieldLot ModelChoiceField, create a choice field with structure (id,printed string)
(("id_lot"),printed string)


DONE- Bulk Electrolytes
    - first, a form with 10 fields of molecule/molecule-lot, and 10 fields of defaults.
    - then, a formset with 10 fields of amounts. (plus proprietary, proprietary name, notes), intialized with the defaults

- cleanup separator
- make drycell functional

DONE - add units to all important quantities.
DONE - add unknown to all important quantities
DONE    - for each quantity, determine if unknown/known/NA is appropriate.
DONE - date widgets

- print the temperature and turbostratic misalignment in a nicer way. (i.e. too many sig. figs.) 

DONE - lots names are now minimal
DONE - make unknown ratios behave the same as unknown stochiometry.
DONE - for things that are normalized to 100%, take the sum of everything except the None,
  and if the total is 0, set to 1 for purposes of renormalization

DONE - remove all the fields that I don't currently care about.
DONE - good name generation
DONE - good uniqueness check.(library)
DONE - good uniqueness check (view)
DONE - test uniqueness check
DONE - streamline the various definitions into much simpler and unique flows.


- create a separate page for:
DONE    - electrolyte definition (molecule, electrolyte)
DONE    - electrode definition (coating, active, inactive, electrode)
DONE    - separator definition (coating, separator material, separator)
DONE    - dry cell definition
DONE    - wet cell definition
- create SOP for entering info (especially incomplete info)
- allow modifications
DONE - allow unknown values
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

