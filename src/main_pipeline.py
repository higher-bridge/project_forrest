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

from utils.file_management import load_merged_files, write_to_tsv
from utils.pipeline_helper import group_by_chunks
from utils.plots import plot_feature_hist, plot_heartrate_hist, plot_heartrate_over_time


if __name__ == '__main__':
    dataframes, IDs = load_merged_files('eyetracking', suffix='*-processed.tsv')

    # Explore (make plots)
    dataframes_grouped = group_by_chunks(dataframes)
    write_to_tsv(dataframes_grouped, IDs, suffix='-grouped.tsv')

    # Combine all dataframes into one big dataframe and plot
    combined_df = pd.concat(dataframes_grouped)
    plot_feature_hist(combined_df)
    plot_heartrate_hist(combined_df)
    plot_heartrate_over_time(combined_df)

    # Pre-process (mean, median, PCA, etc.)

    # Model

    # Plot results

