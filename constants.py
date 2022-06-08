"""
project_forrest
Copyright (C) 2021 Utrecht University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path

import numpy as np
from matplotlib import rcParams

SEED     = 42                         # Set a seed for reproducibility
ROOT_DIR = Path(__file__).parent  # Set root directory for the entire project (can usually be left alone)

N_JOBS   = 4                      # Set number of CPU threads for fixation classification and model parameter search

PX2DEG   = 0.0186                 # Conversion factor from pixels to degrees vis. angle
HZ       = 1000.0                 # Sampling rate of eye tracker
HZ_HEART = 500                    # Sampling rate of pulse oximetry

HESSELS_SAVGOL_LEN  = 31          # Window length of Savitzky-Golay filter in pre-processing (default 31)
HESSELS_THR         = 5000        # Initial slow/fast phase threshold (default 5000)
HESSELS_LAMBDA      = 2.5         # Number of standard deviations (default 2.5)
HESSELS_MAX_ITER    = 200         # Max iterations for threshold adaptation (default 200)
HESSELS_WINDOW_SIZE = 8 * HZ      # Threshold adaptation window (default 8 seconds) * sampling rate
HESSELS_MIN_AMP     = 1.0         # Minimal amplitude of fast candidates for merging slow candidates (default 1.0)
HESSELS_MIN_FIX     = 60          # Minimal fixation duration (default 60)
                                  # MIN_AMP and MIN_FIX come from Hooge, I. T. C., Niehorster, D. C., Nyström, M.,
                                  # Andersson, R., & Hessels, R. S. (2022). Fixation classification: how to merge and
                                  # select fixation candidates. Behavior Research Methods, 2001.
                                  # https://doi.org/10.3758/s13428-021-01723-1)

PURSUIT_AS_FIX = True             # Indicate whether smooth pursuits should be counted as regular fixations
PURSUIT_THR    = 2.0              # Minimal avg_vel to call something a pursuit

CHUNK_SIZE    = 30                # (in seconds). Split up a participant's dataset into chunks
SD_DEV_THRESH = .5                # Set threshold for high/low heartrate label as N standard deviation(s)
DETREND_HR    = False             # Whether to linearly detrend heart rates over the whole viewing

IND_VARS = ['duration', 'amp', 'peak_vel', 'med_vel']  # Independent (predictive) variables
DEP_VAR_BINARY = 'label_hr_median'                     # Dependent (to-predict) variable (for binary classfication)

# Train/test split sizes
TEST_SIZE            = .2
REGRESSION_TEST_SIZE = .2

# Set the range or distribution of hyperparameters for model search
HYPERPARAMS = dict()
HYPERPARAMS['n_estimators'] = list(np.arange(10, 201, step=1))
HYPERPARAMS['max_depth']    = [None] + list(np.arange(1, 31, step=1))
# HYPERPARAMS['max_features'] = [None]

# Model search parameters
HYPERPARAMETER_SAMPLES = 500       # Set how often to sample the hyperparameter distributions in each iteration
SEARCH_ITERATIONS      = 50        # Set how often to re-run the search

# Mind that the following features can be overridden if specified as a main() argument in main_pipeline.py
USE_FEATURE_EXPLOSION  = False    # Whether to retrieve a wide range of statistical features over the data
USE_FEATURE_REDUCTION  = False    # Whether to reduce these back to fewer components with PCA
DIMENSIONS_PER_FEATURE = 2        # Number of components to take from PCA

# Mind that the following feature can also be overridden if specified as a main() argument in main_pipeline.py
REGRESSION_POLY_DEG = 2           # Indicate to which degree to use polynomial regression (1 = No polynomials)

# Matplotlib (plotting) parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 10
