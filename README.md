# how to create the schema.

install dependencies with 
> pip -r requirements.txt

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
- separate the filenames into another file that is never made public 
 