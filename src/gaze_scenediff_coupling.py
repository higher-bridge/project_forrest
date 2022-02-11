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

from typing import List

import numpy as np
import pandas as pd
import pingouin as pg

from constants import SEED
from utils.file_management import load_merged_files
from utils.scenediff_helper import get_scenediffs


def correlate(x: List[float], y: List[float]) -> None:
    pgcorr = pg.corr(x, y)
    print(pgcorr)

    pglr = pg.linear_regression(x, y)
    print(pglr)


def main() -> None:
    np.random.seed(SEED)
    dataframes, IDs = load_merged_files('eyetracking', suffix='*-grouped.tsv')
    heart_rate_df = pd.concat(dataframes).groupby(['chunk']).agg('mean').reset_index()

    diffs_chunked = get_scenediffs()

    differences = list(diffs_chunked['norm_diff'])
    heart_rates = list(heart_rate_df['heartrate'])
    saccades_vel = list(heart_rate_df['peak_vel Saccade'])
    saccades_count = list(heart_rate_df['count Saccade'])

    print('\nCorrelations for scene difference and heart rate')
    correlate(differences, heart_rates)

    print('\nCorrelations for scene difference and saccade peak velocity')
    correlate(differences, saccades_vel)

    print('\nCorrelations for scene difference and saccade count')
    correlate(differences, saccades_count)


main()
