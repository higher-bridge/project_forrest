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
np.random.seed(SEED)

ROOT_DIR = Path(__file__).parent

N_JOBS = 8  # Set how many CPU threads to use for REMODNAV and grid search

# PX2DEG = 0.0185546875
PX2DEG = 0.0142361  # Conversion factor from pixels to degrees vis. angle
HZ = 1000.0  # Sampling rate of eye tracker
HZ_HEART = 500  # Sampling rate of pulse oximetry

CHUNK_SIZE = 30  # In seconds
SD_DEV_THRESH = .75  # Set threshold for high/low heartrate label as N standard deviation(s)

PURSUIT_AS_FIX = False  # Indicate whether smooth pursuits should be counted as regular fixations
IND_VARS = ['duration', 'amp', 'peak_vel', 'avg_vel']  # , 'med_vel'

TEST_SIZE = .20
HYPERPARAMETER_SAMPLES = 1000  # Set how often to sample the hyperparameter distributions in each iteration
SEARCH_ITERATIONS = 50  # Set how often to re-run the grid search
USE_FEATURE_EXPLOSION = False
USE_FEATURE_REDUCTION = False
DIMENSIONS_PER_FEATURE = 2

# String to help with writing files, no need to touch
EXP_RED_STR = f'EXP_{1 if USE_FEATURE_EXPLOSION else 0}_RED_{1 if USE_FEATURE_REDUCTION else 0}'

# Matplotlib params
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 10
