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

from constants import (DEP_VAR_BINARY, REGRESSION_POLY_DEG,
                       USE_FEATURE_EXPLOSION, USE_FEATURE_REDUCTION)
from utils.file_management import load_merged_files, write_to_tsv
from utils.pipeline_helper import (get_scores_and_parameters, group_by_chunks,
                                   run_model_preselection, run_model_search,
                                   run_regression_model,
                                   run_regression_model_per_participant)
from utils.plots import (plot_feature_hist, plot_feature_importance,
                         plot_heartrate_hist, plot_heartrate_over_time)


def run_single_pipeline(group: bool = False,
                        plot: bool = True,
                        process: bool = True,
                        preselection: bool = True,
                        search: bool = True,
                        regression: bool = True,
                        regression_plot: bool = False,
                        regression_per_participant: bool = True,
                        binary_label: str = None,
                        feature_explosion: bool = None,
                        feature_reduction: bool = None,
                        poly_degree: int = None) -> None:
    # Check whether these parameters are specified, otherwise grab them from constants.py
    if feature_explosion is None or feature_reduction is None:
        feature_explosion, feature_reduction = USE_FEATURE_EXPLOSION, USE_FEATURE_REDUCTION
    if poly_degree is None:
        poly_degree = REGRESSION_POLY_DEG
    if binary_label is None:
        binary_label = DEP_VAR_BINARY
    if poly_degree > 2:
        UserWarning('Are you sure you want to use a polynomial regression with degree >2? '
                    'This will increase the number of variables and computational time.')

    dataframes, IDs = load_merged_files('eyetracking', suffix='*-processed.tsv')

    # Group data and write to tsv
    if group:
        dataframes_grouped = group_by_chunks(dataframes, feature_explosion=False, flatten=False)
        write_to_tsv(dataframes_grouped, IDs, suffix='-grouped.tsv')

    # Combine all dataframes into one big dataframe and plot
    if plot:
        dataframes_grouped = group_by_chunks(dataframes, feature_explosion=False, flatten=False)
        combined_df = pd.concat(dataframes_grouped)
        plot_feature_hist(combined_df)
        plot_heartrate_hist(combined_df)
        plot_heartrate_over_time(combined_df, feature='heartrate')

    # Pre-process data
    if not process:
        try:
            dataframes_exploded, IDs = load_merged_files('eyetracking',
                                                         suffix=f'*exploded-EXP{int(feature_explosion)}_RED{int(feature_reduction)}.tsv')
        except:
            print('run_single_pipeline(): "process" was set to False, but no exploded data available. '
                  'Trying with pre-processing...')
            process = True

    if process:
        dataframes_exploded = group_by_chunks(dataframes, feature_explosion=feature_explosion, flatten=True)
        write_to_tsv(dataframes_exploded, IDs, 'eyetracking',
                     f'-exploded-EXP{int(feature_explosion)}_RED{int(feature_reduction)}.tsv')

    # Model
    print(f'Running models with EXPLOSION={feature_explosion}, REDUCTION={feature_reduction}, '
          f'binary label={binary_label}.')
    if preselection:
        run_model_preselection(dataframes_exploded,
                               feature_explosion=feature_explosion,
                               feature_reduction=feature_reduction)

    if search:
        run_model_search(dataframes_exploded,
                         feature_explosion=feature_explosion,
                         feature_reduction=feature_reduction
                         )
        get_scores_and_parameters(feature_explosion=feature_explosion, feature_reduction=feature_reduction)

        # Plot results
        try:
            plot_feature_importance(feature_explosion=feature_explosion, feature_reduction=feature_reduction)
        except Exception as e:
            print(f'Could not plot feature importance (EXP{feature_explosion}, RED{feature_reduction}): {e}')

    if regression:
        run_regression_model(dataframes_exploded,
                             feature_explosion=feature_explosion,
                             feature_reduction=feature_reduction,
                             poly_degree=poly_degree,
                             y_feature='heartrate')

        if regression_per_participant:
            run_regression_model_per_participant(dataframes_exploded, IDs,
                                                 feature_explosion=feature_explosion,
                                                 feature_reduction=feature_reduction,
                                                 poly_degree=poly_degree,
                                                 y_feature='heartrate')


if __name__ == '__main__':
    run_single_pipeline(group=True)
