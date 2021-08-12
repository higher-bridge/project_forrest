# project_forrest

### Procedure
#### Data
- Links to the raw data can be retrieved from https://studyforrest.org (specifically [Hanke et al., 2016](https://www.nature.com/articles/sdata201692)) and should be placed in a subfolder of the ```data``` folder: ```data/eyetracking``` or ```data/heartrate```
- Raw data takes the form of ```sub-02_ses-movie_func_sub-02_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv``` for eyetracking, and ```sub-02_ses-movie_func_sub-02_ses-movie_task-movie_run-1_recording-cardresp_physio.tsv``` for pulse oximetry. 
- The current version of the code looks at characters 5 and 6 to determine participant ID. Data is also sorted based on filename, so ensure that there is some numbering in the filenames for the multiple runs per participant.
#### Code
- After raw data has been placed in the correct folders, run ```src/main_dataloader.py```. This will perform initial pre-processing and fixation/heartrate extraction. This can take a while to run. Any static variables can be changed in ```constants.py```. 
- Then, run ```src/main_pipeline.py```. This has been designed to run separately from the dataloader, so that the dataloader only needs to be run once. The main pipeline can also take some time to run. Again, static variables can be changed in ```constants.py```. 


### Known issues
* Known issue: the Gini impurity plot currently does not work with feature explosion.