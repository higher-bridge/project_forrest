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

from itertools import repeat
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from constants import ROOT_DIR


def get_list_of_files(measurement_type: str, file_type: str = None) -> List[Path]:
    if file_type is None:
        file_type = '*.tsv'

    path = ROOT_DIR / 'data' / measurement_type

    return sorted(path.glob(file_type))


def get_filenames_dict(measurement_type: str, filetype_arg: str = None) -> Dict:
    files = get_list_of_files(measurement_type, filetype_arg)

    file_dict = dict()

    for f in files:
        ID = f.name[4:6]

        try:
            file_dict[ID].append(f)
        except KeyError:
            file_dict[ID] = []
            file_dict[ID].append(f)

    print(f'Found {len(file_dict.keys())} IDs for {measurement_type}: {list(file_dict.keys())}')
    return file_dict


def load_and_concatenate_files(measurement_type: str, filetype_arg: str = None) -> None:
    file_dict = get_filenames_dict(measurement_type, filetype_arg)

    df_dict = dict()

    print(f'Merging all files per ID with type "{measurement_type}"')

    for file_key in list(file_dict.keys()):
        if measurement_type == 'normdiff':  # normdiff.tsv has headers
            dfs = [pd.read_csv(f, sep='\t', encoding='utf-8') for f in file_dict[file_key]]

            for df_prev, df in zip(dfs[:-1], dfs[1:]):
                df.iloc[:, 0] += df_prev.iloc[-1, 0]

        else:
            dfs = [pd.read_csv(f, sep='\t', header=None, encoding='utf-8') for f in file_dict[file_key]]

        df = pd.concat(dfs)

        df_dict[file_key] = df

        # new_path = ROOT_DIR / 'data' / measurement_type / f'{file_key}-{measurement_type}-merged.csv'
        # df.to_csv(new_path,
        #           header=False, index=False)

        new_path = ROOT_DIR / 'data' / measurement_type / f'{file_key}-{measurement_type}-merged.tsv'
        df.to_csv(new_path,
                  header=False, index=False, sep='\t')


def load_merged_files(measurement_type: str, suffix: str = '*-merged.tsv') -> Tuple[List[pd.DataFrame], List[str]]:
    files = get_list_of_files(measurement_type, suffix)

    if len(files) == 0:
        raise Exception(f'No tsv-files of {measurement_type} type! Run load_and_concatenate_files first.')

    IDs = [f.name[:2] for f in files]

    print(f'Loading {len(files)} files of type {measurement_type}: {IDs}')

    if suffix == '*-merged.tsv':
        dfs = [pd.read_csv(f, sep='\t', header=None) for f in files]
    else:
        dfs = [pd.read_csv(f, sep='\t') for f in files]

    return dfs, IDs


def add_ID_column(dfs: List[pd.DataFrame], IDs: List[str]) -> List[pd.DataFrame]:
    new_dfs = []

    for df, ID in zip(dfs, IDs):
        df['ID'] = list(repeat(ID, len(df)))
        new_dfs.append(df)

    return new_dfs


def write_to_tsv(dfs: List[pd.DataFrame], IDs: List[str],
                 measurement_type: str = 'eyetracking', suffix: str = '-processed.tsv') -> None:
    for df, ID in zip(dfs, IDs):
        path = ROOT_DIR / 'data' / measurement_type / f'{ID}-{measurement_type}{suffix}'
        df.to_csv(path, sep='\t')
