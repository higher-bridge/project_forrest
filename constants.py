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

import numpy as np
from pathlib import Path
from matplotlib import rcParams

SEED = 42  # Set a seed for reproducibility

ROOT_DIR = Path(__file__).parent

N_JOBS = 8  # Set how many CPU threads to use for REMODNAV and grid search

PX2DEG = 0.0142361  # Conversion factor from pixels to degrees vis. angle
HZ = 1000.0  # Sampling rate of eye tracker
HZ_HEART = 500  # Sampling rate of pulse oximetry

CHUNK_SIZE = 30  # In seconds
SD_DEV_THRESH = .75  # Set threshold for high/low heartrate label as N standard deviation(s)

PURSUIT_AS_FIX = False  # Indicate whether smooth pursuits should be counted as regular fixations
IND_VARS = ['duration', 'amp', 'peak_vel', 'avg_vel']  # , 'med_vel'

HYPERPARAMS = dict()
HYPERPARAMS['n_estimators'] = list(np.arange(10, 160, step=1))
HYPERPARAMS['max_depth'] = list(np.arange(1, 21, step=1)) + [None]
HYPERPARAMS['max_features'] = list(np.arange(1, 16, step=1))

# Train/test split sizes
TEST_SIZE = .20
REGRESSION_TEST_SIZE = .20

# Model search parameters
HYPERPARAMETER_SAMPLES = 10  # Set how often to sample the hyperparameter distributions in each iteration
SEARCH_ITERATIONS = 3  # Set how often to re-run the grid search

# Mind that the following features can be overridden if specified as a main() argument in main_pipeline.py
USE_FEATURE_EXPLOSION = False
USE_FEATURE_REDUCTION = False

DIMENSIONS_PER_FEATURE = 2

# Mind that the following feature can also be overridden if specified as a main() argument in main_pipeline.py
REGRESSION_POLY_DEG = 2  # Indicate whether to use polynomial regression and to which degree (1 = No polynomials)

# Matplotlib (plotting) parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 10
