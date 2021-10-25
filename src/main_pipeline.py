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
import pandas as pd

from utils.file_management import load_merged_files, write_to_tsv
from utils.pipeline_helper import (get_scores_and_parameters,
                                   group_by_chunks, run_model_preselection,
                                   run_model_search, run_regression_model,
                                   run_regression_model_per_participant)
from utils.plots import (plot_feature_hist, plot_heartrate_hist,
                         plot_heartrate_over_time, plot_gini_coefficients,
                         plot_linear_predictions, plot_linear_predictions_scatter)
from constants import USE_FEATURE_EXPLOSION, USE_FEATURE_REDUCTION, SEED


def main(group: bool = False,
         plot: bool = True,
         process: bool = True,
         preselection: bool = True,
         search: bool = True,
         regression: bool = True) -> None:

    dataframes, IDs = load_merged_files('eyetracking', suffix='*-processed.tsv')

    # Group data and write to tsv
    if group:
        dataframes_grouped = group_by_chunks(dataframes, flatten=False)
        write_to_tsv(dataframes_grouped, IDs, suffix='-grouped.tsv')

    # Combine all dataframes into one big dataframe and plot
    if plot:
        dataframes_grouped = group_by_chunks(dataframes, flatten=False)
        combined_df = pd.concat(dataframes_grouped)
        plot_feature_hist(combined_df)
        plot_heartrate_hist(combined_df)
        plot_heartrate_over_time(combined_df, feature='heartrate')

    # Pre-process data
    if process:
        dataframes_exploded = group_by_chunks(dataframes, flatten=True)
        write_to_tsv(dataframes_exploded, IDs, 'eyetracking', '-exploded.tsv')

    # Or load the pre-processed data if available
    else:
        dataframes_exploded, IDs = load_merged_files('eyetracking', suffix='*-exploded.tsv')

    # Model
    print(f'Running models with EXPLOSION={USE_FEATURE_EXPLOSION}, REDUCTION={USE_FEATURE_REDUCTION}.')
    if preselection:
        run_model_preselection(dataframes_exploded)

    if search:
        run_model_search(dataframes_exploded)
        get_scores_and_parameters()

        # Plot results
        try:
            plot_gini_coefficients()
        except:
            print('Could not plot gini coefficients')

    if regression:
        run_regression_model(dataframes_exploded, y_feature='heartrate')
        plot_linear_predictions_scatter()
        run_regression_model_per_participant(dataframes_exploded, IDs, y_feature='heartrate')


main()
