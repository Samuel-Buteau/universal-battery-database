#TODO

- ~~make a page for the dry cell~~
- ~~make a page for the wet cell~~
- ~~make sure the adding of these are in accordance with the double equality principles.~~
- ~~load the data into the compiled dataset.~~
- create a delete page

----------------------

COMPLETION

----------------------

- create the various modes of definition 
 (override, always produce a valid name, etc.)
- ~~use electrolyte composition in the ML model~~
- make visualization of the embeddings of various molecules.


#TODO

- ~~None means ?~~
- ~~If there is a NULL option (i.e. anode-free, no-coating, etc.. That needs to be defined as a specific object)~~
- ~~is valid means database references are not None~~
- ~~s~~ 


#TODO: delete page
- table of all object kinds.
- multi-select.
- one delete button that deletes everything selected.

#TODO: override

- For all the define pages, have a choice field at the top called "override handle".
- it can be None
- If None, behavior is same as before.
- If not None, 
  - do same checks (if what is entered is invalid, abandon creation.)
  - look for equality but remove override id from search.
  - override the same id.


#TODO: load data in dataset

- ~~create dummy electrodes A, B, C~~
- ~~create dummy electrolytes (not needed)~~
- ~~assign to the wet cells~~
- ~~query the database in compile_dataset~~

- ~~for now, we operate at the level of lots (i.e. two lots of the same type will be different)~~

#TODO: universal normalization

- ~~instead of normalizing things by each cell's max_cap, compute the total max_cap and normalize by that.~~
- ~~make the current grid universal and work exactly like other grids.~~


#TODO: physical grounding of shift

- ~~Feed a capacity tensor to the model (CC)~~
- ~~write a prediction for voltage (CC)~~

- When testing, have a way to:
    - ~~bypass this process with dummy values~~
    - plot the results.
    
There seems to be "out of bound" Q values in the model, caused by very large shift
parameters.

- ~~visualize Vplus, Vminus, V_total for various values of shift. 
(and include out of bound values)~~



With the visualization in place, problems are visible;

- The optimization is unstable (i.e. many runs lead to different outcomes)
  this can be mitigated by going to bigger neural nets, but this seems wasteful for 
  simple things like resistance.
- With a much higher sampling number for the forall penalties, some of the penalities 
  might be too effective. We need to rebalance them to more reasonable values.
  question: if we sample even more, can we reduce further the magnitude of the penalties?
  this might further stabilize training. 
