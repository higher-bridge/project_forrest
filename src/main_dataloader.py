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

from utils.detection import (add_bpm_to_eyetracking, get_bpm_dict,
                             run_remodnav, split_into_chunks)
from utils.file_management import (add_ID_column, get_list_of_files,
                                   load_and_concatenate_files,
                                   load_merged_files, write_to_tsv)


def main() -> None:
    # If not done before, concat files so each ID has one associated file instead of 8
    load_and_concatenate_files('eyetracking', '*physio.tsv')
    load_and_concatenate_files('heartrate', '*physio.tsv')

    # Load the merged files
    df_hr, ID_hr = load_merged_files('heartrate')

    # Retrieve file names. Use if fixation extraction has not been done before
    files_et = get_list_of_files('eyetracking', '*-merged.tsv')
    results = run_remodnav(files_et)

    # Use if extraction already done before
    results, ID_et = load_merged_files('eyetracking', '*-extracted.tsv')

    # Add chunk indices to ET dataframes
    df_et = [split_into_chunks(df, 'eyetracking') for df in results]

    # Get bpm per chunk and append as column to each ET file
    bpm_dict = get_bpm_dict(df_hr, ID_hr, verbose=False)
    df_et = add_bpm_to_eyetracking(df_et, ID_et, bpm_dict)

    # Save ET files
    add_ID_column(df_et, ID_et)
    write_to_tsv(df_et, ID_et)


main()
