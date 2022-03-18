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

from constants import SEED
from utils.detection import (add_bpm_to_eyetracking, get_bpm_dict,
                             run_hessels_classifier, split_into_chunks)
from utils.file_management import (add_ID_column, concatenate_files,
                                   get_list_of_files, load_merged_files,
                                   write_to_tsv)
from utils.plots import plot_phase_distributions


def main() -> None:
    np.random.seed(SEED)

    # Retrieve file names and run fixation detection with (takes a while. Number of CPU cores to use
    # can be specified in constants.py). Use if fixation extraction has not been done before and/or if *-extracted.tsv
    # files are not available:
    files_et = get_list_of_files('eyetracking', '*physio.tsv')
    run_hessels_classifier(files_et)
    plot_phase_distributions(files_et)
    concatenate_files('eyetracking', '*-extracted.tsv')

    # Load oculomotor features
    results, ID_et = load_merged_files('eyetracking', '*-merged.tsv')

    # Add chunk indices to ET dataframes
    df_et = [split_into_chunks(df, 'eyetracking') for df in results]

    # Merge and load heart rate files
    concatenate_files('heartrate', '*physio.tsv')
    df_hr, ID_hr = load_merged_files('heartrate')

    # Get bpm per chunk and append as column to each ET file
    bpm_dict = get_bpm_dict(df_hr, ID_hr, verbose=False)
    df_et = add_bpm_to_eyetracking(df_et, ID_et, bpm_dict)

    # Save ET files
    add_ID_column(df_et, ID_et)
    write_to_tsv(df_et, ID_et)


main()
