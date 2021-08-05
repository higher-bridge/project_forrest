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
from pathlib import Path
from constants import ROOT_DIR


def get_list_of_files(measurement_type, file_type='*.tsv'):
    path = ROOT_DIR / 'data' / measurement_type

    return sorted(path.glob(file_type))


# def get_unique_IDs(files):
#     names = [f.name for f in files]
#
#     IDs = []
#     for f in names:
#         ID = f[4:6]
#         if ID not in IDs:
#             IDs.append(ID)
#
#     print(f'Found {len(IDs)} IDs', IDs)
#     return IDs


def get_filenames_dict(measurement_type):
    files = get_list_of_files(measurement_type)

    file_dict = dict()

    for f in files:
        ID = f.name[4:6]

        try:
            file_dict[ID].append(f)
        except KeyError:
            file_dict[ID] = []
            file_dict[ID].append(f)

    print(f'Found {len(file_dict.keys())} IDs for {measurement_type}')
    return file_dict


def load_and_concatenate_files(measurement_type):
    file_dict = get_filenames_dict(measurement_type)

    df_dict = dict()

    print(f'Merging all files per ID with type "{measurement_type}"')

    for file_key in list(file_dict.keys()):
        dfs = [pd.read_csv(f, sep='\t', header=None, encoding='utf-8') for f in file_dict[file_key]]
        df = pd.concat(dfs)

        df_dict[file_key] = df

        new_path = ROOT_DIR / 'data' / measurement_type / f'{file_key}-{measurement_type}-merged.csv'
        df.to_csv(new_path,
                  header=False, index=False)

        new_path = Path(str(new_path).replace('.csv', '.tsv'))
        df.to_csv(new_path,
                  header=False, index=False, sep='\t')


def load_merged_files(measurement_type, suffix='*-merged.csv'):
    files = get_list_of_files(measurement_type, suffix)

    if len(files) == 0:
        raise Exception(f'No csv-files of {measurement_type} type! Run load_and_concatenate_files first.')

    IDs = [f.name[:2] for f in files]

    print(f'Loading {len(files)} files of type {measurement_type}: {IDs}')

    if suffix == '*-merged.csv':
        dfs = [pd.read_csv(f, header=None) for f in files]
    else:
        dfs = [pd.read_csv(f, sep='\t') for f in files]

    return dfs, IDs


def write_to_tsv(dfs, IDs, measurement_type='eyetracking', suffix='-processed.tsv'):
    for df, ID in zip(dfs, IDs):
        path = ROOT_DIR / 'data' / measurement_type / f'{ID}-{measurement_type}-{suffix}'
        df.to_csv(path, sep='\t')

