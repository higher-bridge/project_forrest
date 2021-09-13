# project_forrest

This repository contains the code and outcome measures for the following manuscript (link will be added once manuscript becomes available):

**Seeing the Forrest through the trees: Random forests predict heart rate based on eye movement features in a naturalistic viewing task.**
Alex J. Hoogerbrugge, Zoril A. Oláh, Edwin S. Dalmaijer, Tanja C.W. Nijboer, Stefan Van der Stigchel


### Procedure
#### Data
- Since this project uses data from a publicly available source, it is not included in the repository.
- Links to the raw data and their descriptions can be retrieved from https://studyforrest.org (specifically [Hanke et al., 2016](https://www.nature.com/articles/sdata201692) for the publication and [OpenNeuro](https://openneuro.org/datasets/ds000113/versions/1.3.0) for the direct link to the data) and should be placed in a subfolder of the ```data``` folder: ```data/eyetracking``` or ```data/heartrate``` respectively.
- Raw data takes the form of ```sub-02_ses-movie_func_sub-02_ses-movie_task-movie_run-1_recording-eyegaze_physio.tsv``` for eyetracking, and ```sub-02_ses-movie_func_sub-02_ses-movie_task-movie_run-1_recording-cardresp_physio.tsv``` for pulse oximetry. Discard all other files.
- The current version of the code looks at characters 5 and 6 to determine participant ID. Data is also sorted based on filename, so ensure that there is some numbering in the filenames for the multiple runs per participant (as is the case in the original naming scheme).
#### Reproducibility
- A seed value of 42 is set for numpy in ```constants.py```.
- All relevant output values, matching those in the manuscript, should be included in the repository. ```EXP_[0/1]_RED[0/1]``` indicate whether feature explosion and dimensionality reduction were applied to achieve this outcome measure (0 = False, 1 = True).
- I have attempted to include the correct environment to exactly reproduce the results. Install with [Anaconda](https://www.anaconda.com/): ```conda install --name ENV_NAME --file environment.yml```.
- However, I run a noMKL environment, so have not been able to test this extensively on different systems and CPUs.
- Nonetheless, the versions of the most important packages (e.g., scikit-learn) are equal. Please let me know if you cannot reproduce the results. 
#### Code
- The ```N_JOBS``` variable in ```constants.py``` sets the number of CPU threads the code will use. Be sure to check whether this is set to a suitable value for your CPU. **Note that setting a different number of jobs will affect  outcomes.**
- After raw data has been placed in the correct folders, run ```src/main_dataloader.py```. This will perform initial pre-processing and fixation/heartrate extraction. This can take a while to run. 
    - Any static variables can be changed in ```constants.py```. 
- Then, run ```src/main_pipeline.py```. This has been designed to run separately from the dataloader, so that the dataloader only needs to be run once. The main pipeline can also take some time to run (depending on ```N_JOBS```, ```HYPERPARAMETER_SAMPLES``` and ```SEARCH_ITERATIONS```). 
    - Again, static variables can be changed in ```constants.py```. 


### Known issues
* The Gini impurity plot currently does not work with feature explosion.