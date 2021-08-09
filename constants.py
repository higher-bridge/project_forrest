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
from matplotlib import rcParams

ROOT_DIR = Path(__file__).parent

N_JOBS_REMODNAV = 8  # Set how many CPU threads to use for REMODNAV (process participants simultaneously)

# PX2DEG = 0.0185546875
PX2DEG = 0.0142361  # Conversion factor from pixels to degrees vis. angle
HZ = 1000.0  # Sampling rate of eye tracker
HZ_HEART = 500  # Sampling rate of pulse oximetry

CHUNK_SIZE = 30
SD_DEV_THRESH = .75  # Set threshold for high/low heartrate label as N standard deviation(s)

PURSUIT_AS_FIX = False  # Indicate whether smooth pursuits should be counted as regular fixations
IND_VARS = ['duration', 'amp', 'peak_vel', 'avg_vel']  # , 'med_vel'

TEST_SIZE = .20
USE_FEATURE_EXPLOSION = True
DIMENSIONS_PER_FEATURE = 2

# Matplotlib params
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 9
