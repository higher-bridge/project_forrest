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
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.svm import SVC
from tpot import TPOTClassifier

from constants import (DIMENSIONS_PER_FEATURE, IND_VARS, PURSUIT_AS_FIX,
                       ROOT_DIR, TEST_SIZE, USE_FEATURE_EXPLOSION, USE_FEATURE_REDUCTION,
                       EXP_RED_STR)
from utils.statistical_features import stat_features


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


def group_by_chunks(dfs: List[pd.DataFrame],
                    feature_explosion: bool = USE_FEATURE_EXPLOSION,
                    flatten: bool = True) -> List[pd.DataFrame]:
    new_dfs = []

    for df in dfs:
        ID = list(df['ID'])[0]

        # Rename eye movement types (e.g. FIXA to Fixation) and remove NA
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

        # Give each movement type its own feature column instead of having one column with strings
        if flatten:
            df_agg['label_hr'] = df_counts['label_hr']['mean']
            df_agg['ID'] = df_counts['ID']['mean']
            df_agg['heartrate'] = df_counts['heartrate']['mean']

            df_agg = flatten_dataframe(df_agg)

        df_agg['label_hr'] = df_agg['label_hr'].apply(rename_labels)
        df_agg['ID'] = list(repeat(ID, len(df_agg)))

        new_dfs.append(df_agg)

    return new_dfs


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    flattened_df = df.pivot_table(columns='label', index='chunk')

    cols_to_unify = []
    replacement_cols = dict()
    for col in flattened_df.columns:
        if col[0] in ['ID', 'heartrate', 'label_hr']:
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


def reduce_dimensionality(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.array, np.array, List[str]]:
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


def prepare_data(df: pd.DataFrame) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], List[str]]:
    # Drop NANs and remove data where label is not high or low
    df = df.dropna()
    df = df.loc[df['label_hr'] != 'normal']

    y_ = list(df['label_hr'])
    y = LabelBinarizer().fit_transform(y_).ravel()
    indices = np.arange(len(y_))

    X_base = df.drop(['ID', 'heartrate', 'label_hr', 'chunk'], axis=1)

    # Retrieve a train/test split
    X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(X_base, y, indices,
                                                                             test_size=TEST_SIZE,
                                                                             stratify=y)

    column_names = X_train.columns

    if USE_FEATURE_EXPLOSION and USE_FEATURE_REDUCTION:
        X_train, X_test, column_names = reduce_dimensionality(X_train, X_test)

    s = StandardScaler()
    X_train_scaled = s.fit_transform(X_train)
    X_test_scaled = s.transform(X_test)

    return (X_train_scaled, X_test_scaled), (y_train, y_test), column_names


def generate_hyperparameters() -> Dict:
    hyperparams = dict()

    hyperparams['n_estimators'] = list(np.arange(10, 160, step=10))
    hyperparams['max_depth'] = list(np.arange(1, 21, step=1)) + [None]
    # hyperparams['min_samples_split'] = list(np.arange(2, 11, step=1))
    # hyperparams['min_samples_leaf'] = list(np.arange(0.1, .6, step=.1))
    hyperparams['max_features'] = list(np.arange(1, 16, step=1))

    return hyperparams


def run_model_search(dataframes: List[pd.DataFrame]) -> None:
    df = pd.concat(dataframes)
    get_data_stats(df)

    (X, X_test), (y, y_test), column_names = prepare_data(df)
    print(f'Train set: {len(X)} values. Test set: {len(X_test)} values.')

    hyperparams = generate_hyperparameters()

    # Run a grid search with the specified hyperparameters
    start = time.time()
    grid_search = GridSearchCV(RandomForestClassifier(),
                               param_grid=hyperparams,
                               scoring='roc_auc',
                               n_jobs=8,
                               verbose=1,
                               return_train_score=True)
    grid_search.fit(X, y)

    # Write and print scores
    to_write = f'Best score: {round(grid_search.best_score_, 3)}. ' \
               f'Score on test set: {round(grid_search.score(X_test, y_test), 3)}. ' \
               f'Duration: {round((time.time() - start) / 60, 2)} minutes.'

    with open(ROOT_DIR / 'results' / f'model_performance_{EXP_RED_STR}.txt', 'w') as wf:
        wf.write(to_write)
    print('\n', to_write, '\n')

    # Retrieve and write results per parameter set
    results = pd.DataFrame(grid_search.cv_results_)
    results['overfit_factor'] = results['mean_train_score'] / results['mean_test_score']
    results.to_csv(ROOT_DIR / 'results' / f'cv_results_{EXP_RED_STR}.csv')

    # Retrieve and write feature importances from the best estimator, if possible
    try:
        importances = grid_search.best_estimator_.feature_importances_
        importances_df = pd.DataFrame(importances, index=column_names).T
        importances_df.to_csv(ROOT_DIR / 'results' / f'best_estimator_importances_{EXP_RED_STR}.csv')
    except:
        print('Model does not have feature importances')

    try:
        params = grid_search.best_estimator_.get_params(deep=False)
        params_df = pd.DataFrame(params, index=[0])
        params_df.to_csv(ROOT_DIR / 'results' / f'best_estimator_parameters_{EXP_RED_STR}.csv')
    except:
        print('Could not obtain parameters from model')

    # Save best estimator to pickle
    pickle.dump(grid_search.best_estimator_, open(ROOT_DIR / 'results' / f'best_estimator_{EXP_RED_STR}.p', 'wb'))


def run_model(dataframes: List[pd.DataFrame]) -> None:
    df = pd.concat(dataframes)
    get_data_stats(df)

    model_names = ['Logistic Regression', 'Random Forest', 'KNN']

    scores_train = {key: [] for key in model_names}
    scores = {key: [] for key in model_names}

    # Do multiple independent runs in order to get a good average performance metric
    for attempt in range(50):
        (X, X_test), (y, y_test), column_names = prepare_data(df)

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

    with open(ROOT_DIR / 'results' / f'model_preliminary_performance_{EXP_RED_STR}.txt', 'w') as wf:
        wf.write(to_write)
    print(to_write, '\n')


def run_model_tpot(dataframes: List[pd.DataFrame]) -> None:
    df = pd.concat(dataframes)
    get_data_stats(df)

    (X, X_test), (y, y_test), column_names = prepare_data(df)

    pipeline_optimizer = TPOTClassifier(generations=20, scoring='roc_auc',
                                        config_dict='TPOT light',
                                        verbosity=2, n_jobs=8)
    pipeline_optimizer.fit(X, y)

    print(pipeline_optimizer.score(X_test, y_test))

    pipeline_optimizer.export(ROOT_DIR / 'results' / f'tpot_pipeline_export_{EXP_RED_STR}.py')
