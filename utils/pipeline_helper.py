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

import pandas as pd
import numpy as np
from constants import PURSUIT_AS_FIX, IND_VARS, DEP_VAR


def rename_types(x):
    if x == 'FIXA':
        return 'Fixation'
    elif x == 'SACC':
        return 'Saccade'
    elif x == 'PURS':
        return 'Fixation' if PURSUIT_AS_FIX else 'Pursuit'
    else:
        return 'NA'


def group_by_chunks(dfs):
    new_dfs = []

    for df in dfs:
        # Drop first column if necessary
        # df = df.drop(['Unnamed: 0'], axis=1) if 'Unnamed: 0' in df.columns else df

        # Rename eye movement types (e.g. FIXA to Fixation) and remove NA
        df['label'] = df['label'].apply(rename_types)
        df = df.loc[df['label'] != 'NA']

        # Compute a counter and compute the mean
        df_counts = df.groupby(['chunk', 'label']).agg('count').reset_index()
        df_mean = df.groupby(['chunk', 'label']).agg('mean').reset_index()

        # Create a list of unnecessary columns
        columns_to_use = IND_VARS + ['chunk', 'label', 'label_hr']
        drop_columns = [c for c in df.columns if c not in columns_to_use]

        # Drop unnecessary columns from df_mean and append the counter for each movement type
        df_agg = df_mean.drop(drop_columns, axis=1)
        df_agg['count'] = df_counts['onset']

        new_dfs.append(df_agg)

    return new_dfs


