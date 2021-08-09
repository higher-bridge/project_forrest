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
import numpy as np
from itertools import repeat

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

from tpot import TPOTClassifier

from constants import PURSUIT_AS_FIX, IND_VARS, USE_FEATURE_EXPLOSION, TEST_SIZE, ROOT_DIR
from utils.statistical_features import stat_features


def rename_types(x):
    if x == 'FIXA':
        return 'Fixation'
    elif x == 'SACC':
        return 'Saccade'
    elif x == 'PURS':
        return 'Fixation' if PURSUIT_AS_FIX else 'Pursuit'
    else:
        return 'NA'


def rename_labels(x):
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


def rename_features(x):
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


def group_by_chunks(dfs, feature_explosion=USE_FEATURE_EXPLOSION, flatten=True):
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

        if not feature_explosion:
            df_agg = df.groupby(['chunk', 'label']).agg({feat: np.nanmean for feat in IND_VARS}).reset_index()
        else:
            df_agg = df.groupby(['chunk', 'label']).agg({feat: stat_features for feat in IND_VARS}).reset_index()

        df_agg['count'] = df_counts['duration']['count']

        if flatten:
            df_agg['label_hr'] = df_counts['label_hr']['mean']
            df_agg['ID'] = df_counts['ID']['mean']
            df_agg['heartrate'] = df_counts['heartrate']['mean']

            df_agg = flatten_dataframe(df_agg)

        df_agg['label_hr'] = df_agg['label_hr'].apply(rename_labels)
        df_agg['ID'] = list(repeat(ID, len(df_agg)))

        new_dfs.append(df_agg)

    return new_dfs


def flatten_dataframe(df):
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


def get_data_stats(df):
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


def prepare_data(df):
    df = df.dropna()
    df = df.loc[df['label_hr'] != 'normal']

    y_ = list(df['label_hr'])
    y = LabelBinarizer().fit_transform(y_).ravel()
    indices = np.arange(len(y_))

    X_base = df.drop(['ID', 'heartrate', 'label_hr', 'chunk'], axis=1)

    X_train, X_test, y_train, y_test, train_ind, test_ind = train_test_split(X_base, y, indices,
                                                                             test_size=TEST_SIZE,
                                                                             stratify=y)

    # TODO: implement feature reduction in case of feature explosion

    s = StandardScaler()
    X_train_scaled = s.fit_transform(X_train)
    X_test_scaled = s.transform(X_test)

    return (X_train_scaled, X_test_scaled), (y_train, y_test)


def run_model(dataframes):
    df = pd.concat(dataframes)
    get_data_stats(df)

    scores_train = []
    scores = []
    for attempt in range(100):
        (X, X_test), (y, y_test) = prepare_data(df)

        if attempt == 0:
            print(f'Train set: {len(X)} values. Test set: {len(X_test)} values.')

        model = LogisticRegression()
        model.fit(X, y)

        # Performance on train set
        y_pred_prob = model.predict_proba(X)[::, 1]
        auc_ = roc_auc_score(y, y_pred_prob)
        scores_train.append(auc_)

        # Performance on test set
        y_pred_prob = model.predict_proba(X_test)[::, 1]
        auc_ = roc_auc_score(y_test, y_pred_prob)
        scores.append(auc_)


    print(f'Training mean score = {round(np.mean(scores_train), 3)} (SD = {round(np.std(scores_train), 3)})')
    print(f'Testing mean score  = {round(np.mean(scores), 3)} (SD = {round(np.std(scores), 3)})')


def run_model_tpot(dataframes):
    df = pd.concat(dataframes)
    get_data_stats(df)

    (X, X_test), (y, y_test) = prepare_data(df)

    pipeline_optimizer = TPOTClassifier(scoring='roc_auc',
                                        verbosity=2, n_jobs=8)
    pipeline_optimizer.fit(X, y)

    print(pipeline_optimizer.score(X_test, y_test))

    pipeline_optimizer.export(ROOT_DIR / 'results' / 'tpot_pipeline_export.py')


