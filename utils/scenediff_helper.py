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

from utils.file_management import load_merged_files, load_and_concatenate_files
from utils.detection import split_into_chunks


def get_scenediffs() -> pd.DataFrame:
    load_and_concatenate_files('normdiff', '*_normdiff.tsv')
    diffs_list, _ = load_merged_files('normdiff', suffix='*-merged.tsv')

    diffs_list[0].columns = ['onset', 'duration', 'norm_diff']
    diffs = split_into_chunks(diffs_list[0], 'normdiff')

    diffs_chunked = diffs.groupby(['chunk']).agg('mean').reset_index()
    diffs_chunked = diffs_chunked.iloc[:239]

    return diffs_chunked