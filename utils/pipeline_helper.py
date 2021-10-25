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

import pickle
import random
import time
from itertools import repeat
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler, PolynomialFeatures
from sklearn.utils import shuffle

from constants import (DIMENSIONS_PER_FEATURE, HYPERPARAMS, IND_VARS,
                       PURSUIT_AS_FIX, ROOT_DIR, SEARCH_ITERATIONS, TEST_SIZE,
                       N_JOBS, HYPERPARAMETER_SAMPLES,
                       SEED, REGRESSION_TEST_SIZE)
from utils.statistical_features import stat_features
from utils.scenediff_helper import get_scenediffs


def rename_types(x: str) -> str:
    if x == 'FIXA':
        return 'Fixation'
    elif x == 'SACC':
        return 'Saccade'
    elif x == 'PURS':
        return 'Fixation' if PURSUIT_AS_FIX else 'Pursuit'
    else:
        return 'NA'


def rename_labels(x) -> str:
    if np.isnan(x):
        return x

    elif int(x) == -1:
        return 'low'
    elif int(x) == 1:
        return 'high'
    elif int(x) == 0:
        return 'normal'
    else:
        return 'NA'


def rename_features(x: str) -> str:
    if x == 'duration':
        return 'Duration (s)'
    elif x == 'amp':
        return 'Amplitude (°)'
    elif x == 'peak_vel':
        return 'Peak velocity (°/s)'
    elif x == 'med_vel':
        return 'Median velocity (°/s)'
    elif x == 'avg_vel':
        return 'Mean velocity (°/s)'
    elif x == 'count':
        return 'Count (per chunk)'
    else:
        return x


def group_by_chunks(dfs: List[pd.DataFrame], feature_explosion: bool, flatten: bool = True) -> List[pd.DataFrame]:
    # scene_diffs = list(get_scenediffs()['norm_diff'])

    new_dfs = []

    for df in dfs:
        ID = list(df['ID'])[0]

        # Rename eye movement types (e.g. FIXA to Fixation) and remove NA
        if 'FIXA' in list(df['label'].unique()):
            df['label'] = df['label'].apply(rename_types)

        df = df.loc[df['label'] != 'NA']

        # Drop data where heartrate = 0 (usually last chunk)
        df = df.loc[df['heartrate'] > 0.0]

        # Drop unnecessary columns
        df = df.drop(['Unnamed: 0', 'onset', 'start_x', 'start_y', 'end_x', 'end_y'], axis=1)

        # Compute a counter and compute the mean
        df_counts = df.groupby(['chunk', 'label']).agg(['count', 'mean']).reset_index()

        # Aggregate, either by mean or with statistical descriptors
        if not feature_explosion:
            df_agg = df.groupby(['chunk', 'label']).agg({feat: np.nanmean for feat in IND_VARS}).reset_index()
        else:
            df_agg = df.groupby(['chunk', 'label']).agg({feat: stat_features for feat in IND_VARS}).reset_index()

        # Add the count variable again
        df_agg['count'] = df_counts['duration']['count']

        df_agg['label_hr'] = df_counts['label_hr']['mean']
        df_agg['diff_hr'] = df_counts['diff_hr']['mean']
        df_agg['sd_hr'] = df_counts['sd_hr']['mean']
        df_agg['ID'] = df_counts['ID']['mean']
        df_agg['heartrate'] = df_counts['heartrate']['mean']

        # Give each movement type its own feature column instead of having one column with strings
        if flatten:
            df_agg = flatten_dataframe(df_agg)

        df_agg['label_hr'] = df_agg['label_hr'].apply(rename_labels)
        df_agg['ID'] = list(repeat(ID, len(df_agg)))
        # df_agg['scene_diff'] = scene_diffs

        new_dfs.append(df_agg)

    return new_dfs


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    flattened_df = df.pivot_table(columns='label', index='chunk')

    cols_to_unify = []
    replacement_cols = dict()
    for col in flattened_df.columns:
        if col[0] in ['ID', 'heartrate', 'label_hr', 'diff_hr', 'sd_hr']:
            cols_to_unify.append(col)

            if col[0] not in list(replacement_cols.keys()):
                replacement_cols[col[0]] = flattened_df[col].values

    flattened_df = flattened_df.drop(cols_to_unify, axis=1)

    old_columns = flattened_df.columns
    try:
        flattened_df.columns = [f'{c[0]} {c[1]} {c[2]}' for c in old_columns]
    except IndexError:
        flattened_df.columns = [f'{c[0]} {c[1]}' for c in old_columns]

    flattened_df['chunk'] = flattened_df.index

    for key in list(replacement_cols.keys()):
        flattened_df[key] = replacement_cols[key]

    return flattened_df


def get_data_stats(df: pd.DataFrame) -> None:
    len1 = len(df)

    df = df.dropna()
    len2 = (len(df))

    df = df.loc[df['label_hr'] != 'normal']

    len_low = len(df.loc[df['label_hr'] == 'low'])
    perc_low = round((len_low / len2) * 100, 2)
    len_high = len(df.loc[df['label_hr'] == 'high'])
    perc_high = round((len_high / len2) * 100, 2)

    print(f'Total: {len1}. {len1 - len2} NaNs dropped.',
          f'{len_low} low ({perc_low}%), {len_high} high ({perc_high}%).')


def reduce_dimensionality(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.array,
                                                                                np.array,
                                                                                List[str]]:
    # Split dfs and run PCA per feature
    split_dfs = []

    column_names = []

    # Columns are in format "feature descriptor move_type", e.g., "duration nanmean Fixation".
    # For each feature and move_type combination, run separate PCA over all of its descriptors. Then apply the PCA
    # to the test set too.
    for move_type in ['Fixation', 'Pursuit', 'Saccade']:
        for feature in IND_VARS:
            cols_to_use = [col for col in X_train.columns if move_type in col and feature in col]

            df_section_train = X_train.loc[:, cols_to_use]
            df_section_test = X_test.loc[:, cols_to_use]

            pca = PCA(n_components=DIMENSIONS_PER_FEATURE, svd_solver='full', whiten=True)
            pc_train = pca.fit_transform(df_section_train)
            pc_test = pca.transform(df_section_test)

            # Append each column of the array separately
            for i in range(DIMENSIONS_PER_FEATURE):
                split_dfs.append((pc_train[:, i], pc_test[:, i]))
                column_names.append(f'{move_type} {feature} PC{i}')

        # Count variable only has one column per move type, so don't do PCA and just append it again afterward
        split_dfs.append((X_train.loc[:, f'count  {move_type}'],
                          X_test.loc[:, f'count  {move_type}']))
        column_names.append(f'{move_type} count PC1')

    # Now put everything back in a single array, with rows for chunks and columns for PCA features
    X_train_new = np.array([x[0] for x in split_dfs]).T
    X_test_new = np.array([x[1] for x in split_dfs]).T

    return X_train_new, X_test_new, column_names


def remove_outliers(x: np.array, y: List[Any]) -> Tuple[np.array, np.array]:
    num_rows = x.shape[0]
    for i in range(x.shape[1]):
        # Select column
        x_ = x[:, i]

        # Get tenth and ninetieth percentiles
        low = scoreatpercentile(x_, 1)
        high = scoreatpercentile(x_, 99)

        # Find where data in x_ is outside percentiles and set those to np.nan
        mask = np.where((x_ < low) | (x_ > high))
        x_[mask] = np.nan

        # Now fill the column in the original array with the new data
        x[:, i] = x_

    # Find indices where at least one value is NaN and drop from x and y
    keep_rows = ~np.isnan(x).any(axis=1)
    x = x[keep_rows]
    y = np.array(y)[keep_rows]

    # dropped_perc = round((1 - x.shape[0] / num_rows) * 100, 2)
    # print(f'Outlier removal dropped {dropped_perc}% of data')

    return x, y


def prepare_data(df: pd.DataFrame,
                 feature_explosion: bool, feature_reduction: bool) -> Tuple[Tuple[np.array, np.array],
                                                                            Tuple[np.array, np.array],
                                                                            List[str]]:
    # Drop NANs and remove data where label is not high or low
    df = df.dropna()
    df = df.loc[df['label_hr'] != 'normal']

    y_ = list(df['label_hr'])
    X_base = df.drop(['ID', 'heartrate', 'label_hr', 'diff_hr', 'sd_hr', 'chunk'], axis=1)
    column_names = X_base.columns

    X_base, y_ = remove_outliers(np.array(X_base), y_)
    y = LabelBinarizer().fit_transform(y_).ravel()
    indices = np.arange(len(y_))

    # Retrieve a train/test split
    X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(X_base, y, indices,
                                                                             test_size=TEST_SIZE,
                                                                             stratify=y)

    if feature_explosion and feature_reduction:
        X_train = pd.DataFrame(X_train, columns=column_names)
        X_test = pd.DataFrame(X_test, columns=column_names)
        X_train, X_test, column_names = reduce_dimensionality(X_train, X_test)

    s = StandardScaler()
    X_train_scaled = s.fit_transform(X_train)
    X_test_scaled = s.transform(X_test)

    return (X_train_scaled, X_test_scaled), (y_train, y_test), column_names


def prepare_data_continuous(df: pd.DataFrame,
                            feature_explosion: bool, feature_reduction: bool,
                            poly_degree: int,
                            y_feature: str = 'heartrate') -> Tuple[Tuple[np.array, np.array],
                                                                   Tuple[np.array, np.array],
                                                                   List[str]]:
    # Drop NANs
    df = df.dropna()

    # Get heartrates
    y = list(df[y_feature])

    X_base = df.drop(['ID', 'heartrate', 'label_hr', 'diff_hr', 'sd_hr', 'chunk'], axis=1)
    column_names = list(X_base.columns)

    X = np.array(X_base)
    X, y = remove_outliers(X, y)

    # Retrieve a train/test split. We can't use sklearn's train_test_split because y is a continuous variable
    indices = list(np.arange(len(y)))
    indices_shuffled = shuffle(indices)

    train_samples = int((1 - REGRESSION_TEST_SIZE) * len(y))
    train_ind = indices_shuffled[0:train_samples]
    test_ind = indices_shuffled[train_samples:-1]

    X_train, X_test = X[train_ind], X[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]

    if feature_explosion and feature_reduction:
        X_train = pd.DataFrame(X_train, columns=column_names)
        X_test = pd.DataFrame(X_test, columns=column_names)
        X_train, X_test, column_names = reduce_dimensionality(X_train, X_test)

    if poly_degree > 1:
        pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_train = pf.fit_transform(X_train)
        X_test = pf.transform(X_test)
    # else:
    #     s = StandardScaler()
    #     X_train = s.fit_transform(X_train)
    #     X_test = s.transform(X_test)

    return (X_train, X_test), (y_train, y_test), column_names


def run_model_search_iteration(df: pd.DataFrame, iteration_nr: int,
                               feature_explosion: bool, feature_reduction: bool) -> Tuple[RandomizedSearchCV,
                                                                                          float,
                                                                                          str,
                                                                                          List[str]]:
    print(f'\nIteration {iteration_nr + 1}:')
    (X, X_test), (y, y_test), column_names = prepare_data(df,
                                                          feature_explosion=feature_explosion,
                                                          feature_reduction=feature_reduction)

    # Run a grid search with the specified hyperparameters
    start = time.time()
    param_search = RandomizedSearchCV(RandomForestClassifier(),
                                      param_distributions=HYPERPARAMS,
                                      n_iter=HYPERPARAMETER_SAMPLES,
                                      scoring='roc_auc',
                                      n_jobs=N_JOBS,
                                      verbose=1,
                                      return_train_score=True)

    param_search.fit(X, y)
    test_score = param_search.score(X_test, y_test)

    # Print scores
    to_write = f'Best score: {round(param_search.best_score_, 3)}. ' \
               f'Score on test set: {round(test_score, 3)}. ' \
               f'Duration: {round((time.time() - start) / 60, 2)} minutes.\n'
    print(to_write)

    return param_search, test_score, to_write, column_names


def run_model_search(dataframes: List[pd.DataFrame],
                     feature_explosion: bool, feature_reduction: bool) -> None:
    np.random.seed(SEED)
    df = pd.concat(dataframes)
    get_data_stats(df)

    best_estimators = []
    test_scores = []
    cv_scores = []
    text_results = ''
    cv_results = pd.DataFrame()
    gini_coefficients = pd.DataFrame()

    for i in range(SEARCH_ITERATIONS):
        grid_search, test_score, to_write, column_names = run_model_search_iteration(df, i,
                                                                                     feature_explosion=feature_explosion,
                                                                                     feature_reduction=feature_reduction,
                                                                                     )

        best_estimators.append(grid_search.best_estimator_)
        test_scores.append(test_score)
        cv_scores.append(grid_search.best_score_)
        text_results += to_write

        cv_results_iteration = pd.DataFrame(grid_search.cv_results_)
        cv_results_iteration['Iteration'] = [i] * len(cv_results_iteration)
        cv_results = cv_results.append(cv_results_iteration)

        gini_coefficients_iteration = pd.DataFrame(grid_search.best_estimator_.feature_importances_,
                                                   index=column_names).T
        gini_coefficients = gini_coefficients.append(gini_coefficients_iteration)

    # Print mean score on test sets
    print(f'Mean overall CV score on {SEARCH_ITERATIONS} iterations:',
          f'{round(np.mean(cv_scores), 3)} (SD = {round(np.std(cv_scores), 3)})',
          f'Mean overall test score:',
          f'{round(np.mean(test_scores), 3)} (SD = {round(np.std(test_scores), 3)})')

    # Write performance in text
    with open(ROOT_DIR / 'results' / f'model_performance_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.txt', 'w') as wf:
        wf.write(text_results)

    # Save cross-validation results
    cv_results['overfit_factor'] = cv_results['mean_train_score'] / cv_results['mean_test_score']
    cv_results.to_csv(ROOT_DIR / 'results' / f'cv_results_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.csv')

    # Save the gini coefficients
    gini_coefficients.to_csv(ROOT_DIR / 'results' / f'best_estimator_importances_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.csv')

    # Save best estimators to pickle
    pickle.dump(best_estimators, open(ROOT_DIR / 'results' / f'best_estimator_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.p', 'wb'))


def run_model_preselection(dataframes: List[pd.DataFrame],
                           feature_explosion: bool, feature_reduction: bool) -> None:
    np.random.seed(SEED)
    df = pd.concat(dataframes)
    get_data_stats(df)

    model_names = ['Logistic Regression', 'Random Forest', 'KNN']

    scores_train = {key: [] for key in model_names}
    scores = {key: [] for key in model_names}

    # Do multiple independent runs in order to get a good average performance metric
    for attempt in range(50):
        (X, X_test), (y, y_test), column_names = prepare_data(df,
                                                              feature_explosion=feature_explosion,
                                                              feature_reduction=feature_reduction)

        if attempt == 0:
            print(f'Train set: {len(X)} values. Test set: {len(X_test)} values.')

        models = [LogisticRegression(),
                  RandomForestClassifier(),
                  KNeighborsClassifier()]

        for model, model_name in zip(models, model_names):
            model.fit(X, y)

            # Performance on train set
            y_pred_prob = model.predict_proba(X)[::, 1]
            auc_ = roc_auc_score(y, y_pred_prob)
            scores_train[model_name].append(auc_)

            # Performance on test set
            y_pred_prob = model.predict_proba(X_test)[::, 1]
            auc_ = roc_auc_score(y_test, y_pred_prob)
            scores[model_name].append(auc_)

    to_write = str()
    for model_name in model_names:
        train_scores = scores_train[model_name]
        test_scores = scores[model_name]

        to_write += f'\n{model_name}: '
        to_write += f'Training mean score = {round(np.mean(train_scores), 3)} (SD = {round(np.std(train_scores), 3)}). '
        to_write += f'Testing mean score  = {round(np.mean(test_scores), 3)} (SD = {round(np.std(test_scores), 3)}).'

    with open(ROOT_DIR / 'results' / f'model_preliminary_performance_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.txt', 'w') as wf:
        wf.write(to_write)
    print(to_write, '\n')


def get_scores_and_parameters(feature_explosion: bool, feature_reduction: bool,) -> None:
    path = ROOT_DIR / 'results' / f'cv_results_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.csv'
    df = pd.read_csv(path)

    num_iterations = list(df['Iteration'].unique())

    for rank in range(3):
        cv_list = []
        trees_list = []
        depth_list = []
        feats_list = []

        for i in num_iterations:
            df_ = df.loc[df['Iteration'] == i]
            df_ = df_.sort_values(by='mean_test_score', ascending=False)

            best_row = df_.iloc[rank]

            cv_list.append(best_row['mean_test_score'])
            trees_list.append(best_row['param_n_estimators'])
            depth_list.append(best_row['param_max_depth'])
            feats_list.append(best_row['param_max_features'])

        print(f'Rank {rank}:',
              f'Score {round(np.mean(cv_list), 3)}',
              f'Trees {round(np.mean(trees_list), 3)}',
              f'Depth {round(np.nanmean(depth_list), 3)}',
              f'Feat {round(np.mean(feats_list), 3)}')


def run_regression_model(dataframes: List[pd.DataFrame],
                         feature_explosion: bool, feature_reduction: bool,
                         poly_degree: int,
                         y_feature: str = 'heartrate') -> None:
    np.random.seed(SEED)
    df = pd.concat(dataframes)
    get_data_stats(df)

    train_r2 = []
    test_r2 = []
    coefs = []

    best_model = [None, None, None, None, None]
    best_model_score = -1e10

    # Do multiple independent runs in order to get a good average performance metric
    for attempt in range(50):
        (X, X_test), (y, y_test), column_names = prepare_data_continuous(df,
                                                                         feature_explosion=feature_explosion,
                                                                         feature_reduction=feature_reduction,
                                                                         poly_degree=poly_degree,
                                                                         y_feature=y_feature)

        model = LinearRegression()
        model.fit(X, y)
        test_score = model.score(X_test, y_test)

        train_r2.append(model.score(X, y))
        test_r2.append(test_score)
        coefs.append(model.coef_)

        if test_score > best_model_score:
            best_model_score = test_score
            best_model = [model, X_test, y_test, column_names, round(test_score, 2)]

    r2_train = round(np.mean(train_r2), 2)
    r2_test = round(np.mean(test_r2), 2)
    r2_train_sd = round(np.std(train_r2), 2)
    r2_test_sd = round(np.std(test_r2), 2)

    to_write = f'Linear regression (50 runs) mean R-squared on train set = {r2_train} (SD = {r2_train_sd}). ' \
               f'Mean R-squared on test set = {r2_test} (SD = {r2_test_sd}). ' \
               f'Best = {round(best_model_score, 2)}.'
    with open(ROOT_DIR / 'results' / f'linear_estimator_performance_POLYDEG_{poly_degree}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.txt',
              'w') as wf:
        wf.write(to_write)
    print(to_write, '\n')

    pickle.dump(best_model,
                open(ROOT_DIR / 'results' / f'linear_estimator_POLYDEG_{poly_degree}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.p',
                     'wb'))


def run_regression_model_per_participant(dataframes: List[pd.DataFrame], IDs,
                                         feature_explosion: bool, feature_reduction: bool,
                                         poly_degree: int,
                                         y_feature: str = 'heartrate') -> None:
    np.random.seed(SEED)

    ID_dict = {ID: dict() for ID in IDs}
    write_text = ''

    for df, ID in zip(dataframes, IDs):
        train_r2 = []
        test_r2 = []
        coefs = []

        best_model = [None, None, None, None, None]
        best_model_score = -1e10

        # Do multiple independent runs in order to get a good average performance metric
        for attempt in range(50):
            (X, X_test), (y, y_test), column_names = prepare_data_continuous(df,
                                                                             feature_explosion=feature_explosion,
                                                                             feature_reduction=feature_reduction,
                                                                             poly_degree=poly_degree,
                                                                             y_feature=y_feature)

            model = LinearRegression()
            model.fit(X, y)
            test_score = model.score(X_test, y_test)

            train_r2.append(model.score(X, y))
            test_r2.append(test_score)
            coefs.append(model.coef_)

            if test_score > best_model_score:
                best_model_score = test_score
                best_model = [model, X_test, y_test, column_names, round(test_score, 2)]

        r2_train = round(np.mean(train_r2), 2)
        r2_test = round(np.mean(test_r2), 2)
        r2_train_sd = round(np.std(train_r2), 2)
        r2_test_sd = round(np.std(test_r2), 2)
        train_samples = len(y)
        test_samples = len(y_test)

        to_write = f'\n{ID}:' \
                   f'Linear regression (50 runs) mean R-squared on train set = {r2_train} (SD = {r2_train_sd}). ' \
                   f'Mean R-squared on test set = {r2_test} (SD = {r2_test_sd}). ' \
                   f'Best = {round(best_model_score, 2)}.' \
                   f'({train_samples}/{test_samples} train/test samples)'
        write_text += to_write
        print(to_write)

        ID_dict[ID]['Best model'] = best_model
        ID_dict[ID]['R2 train mean'] = r2_train
        ID_dict[ID]['R2 train sd'] = r2_train_sd
        ID_dict[ID]['R2 test mean'] = r2_test
        ID_dict[ID]['R2 test sd'] = r2_test_sd

    regr_stats = compute_regression_stats(poly_degree, feature_explosion, feature_reduction, ID_dict)
    write_text += regr_stats

    with open(
            ROOT_DIR / 'results' / f'linear_estimator_performance_per_participant_POLYDEG_{poly_degree}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.txt',
            'w') as wf:
        wf.write(write_text)

    pickle.dump(ID_dict,
                open(
                    ROOT_DIR / 'results' / f'linear_estimator_per_participant_POLYDEG_{poly_degree}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.p',
                    'wb')
                )


def compute_regression_stats(poly_degree: int,
                             feature_explosion: bool, feature_reduction: bool,
                             ID_dict: Dict[str, Dict[str, Any]] = None) -> str:
    if ID_dict is None:
        ID_dict = pickle.load(open(
            ROOT_DIR / 'results' / f'linear_estimator_per_participant_POLYDEG_{poly_degree}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.p',
            'wb'))

    means_train = []
    means_test = []

    for ID in list(ID_dict.keys()):
        idd = ID_dict[ID]
        means_train.append(idd['R2 train mean'])
        means_test.append(idd['R2 test mean'])

    mean_train = round(np.mean(means_train), 2)
    mean_test = round(np.mean(means_test), 2)

    to_write = f'\nMean R-squared on training = {mean_train}, testing = {mean_test}'
    print(to_write)

    return to_write
