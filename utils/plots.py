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

from pathlib import Path
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from scipy.stats import ttest_1samp

from constants import DEP_VAR_BINARY, IND_VARS, ROOT_DIR, SD_DEV_THRESH, HESSELS_WINDOW_SIZE
from utils.pipeline_helper import rename_features


def plot_phase_distributions(files_et: List[Path]) -> None:
    files = [str(f).replace('.tsv', '-extracted.tsv') for f in files_et]
    dfs = [pd.read_csv(f, delimiter='\t') for f in files]
    df = pd.concat(dfs)
    df = df.loc[df['label'] != 'none']

    for feat in ['duration', 'amp', 'avg_vel', 'med_vel', 'peak_vel']:
        plt.figure()
        sns.histplot(x=feat, data=df, hue='label',
                     multiple='layer', stat='count', element='step')
        plt.xlabel(rename_features(feat))
        # plt.legend(title='Phase type')

        plt.tight_layout()

        save_path = ROOT_DIR / 'results' / 'plots' / f'phase_hist_{feat}_window_{HESSELS_WINDOW_SIZE}.png'
        plt.savefig(save_path, dpi=600)

        plt.show()


def plot_timeseries(x: np.ndarray, x_after: np.ndarray, f: str, smoothing: str) -> None:
    p = plt.figure()
    plt.plot(np.arange(len(x)), x, label='before filter',
             color='black', linewidth=.7)
    plt.plot(np.arange(len(x)), x_after, label=smoothing,
             color='red', linewidth=.7)

    plt.xlabel('Time (ms)')
    plt.ylabel('x')

    plt.legend()
    plt.title(f, fontsize=8)

    plt.tight_layout()
    plt.show()
    plt.close(p)


def plot_feature_hist(df: pd.DataFrame) -> None:
    features = IND_VARS + ['count']
    move_types = list(df['label'].unique())

    nrows = len(features)
    ncols = len(move_types)

    f = plt.figure(figsize=(7.5, 1.35 * nrows))
    axes = [f.add_subplot(nrows, ncols, x + 1) for x in range(nrows * ncols)]

    palette = sns.color_palette('tab10')

    i = 0
    for row, feature in enumerate(features):
        for col, move_type in enumerate(move_types):
            df_move_type = df.loc[df['label'] == move_type]
            df_low = df_move_type.loc[df_move_type[DEP_VAR_BINARY] == 'low']
            df_high = df_move_type.loc[df_move_type[DEP_VAR_BINARY] == 'high']

            sns.kdeplot(x=feature, data=df_high, ax=axes[i],
                        label='High', color=palette[1],
                        fill=True, linewidth=2.5,
                        common_norm=False, linestyle='--',
                        clip=(0.0, 350))
            sns.kdeplot(x=feature, data=df_low, ax=axes[i],
                        label='Low', color=palette[0],
                        fill=True, linewidth=2.5,
                        common_norm=False, linestyle='-',
                        clip=(0.0, 350))

            # Remove y-ticks
            axes[i].set_yticks(list())

            # Set feature label only on first column
            if i % ncols == 0:
                axes[i].set_ylabel(rename_features(feature), fontsize=9)
            else:
                axes[i].set_ylabel('')

            # Set movement type label only beneath last row
            if i >= (nrows * ncols) - ncols:
                axes[i].set_xlabel(f'{move_type}s')
            else:
                axes[i].set_xlabel('')

            # Set legend only in top-left panel
            if i == 0:
                axes[i].legend(fontsize=10, title='Heart rate', loc='best')

            if i < ncols:
                print(f'{move_type}, {len(df_low)} low, {len(df_high)} high,',
                      f'{len(df_move_type) - len(df_low) - len(df_high)} in between. {len(df_move_type)} total.')

            i += 1

    plt.tight_layout()
    save_path = ROOT_DIR / 'results' / 'plots' / f'feature_hist.png'
    plt.savefig(save_path, dpi=600)
    plt.show()


def plot_heartrate_hist(df: pd.DataFrame) -> None:
    df['Heart rate'] = df[DEP_VAR_BINARY]

    f = plt.figure(figsize=(7.5, 10))
    axes = [f.add_subplot(8, 2, i + 1) for i in range(len(list(df['ID'].unique())))]

    for i, ID in enumerate(list(df['ID'].unique())):
        df_ = df.loc[df['ID'] == ID]

        sns.histplot(x='heartrate', data=df_, ax=axes[i], hue='Heart rate',
                     stat='count', multiple='layer', discrete=False, element='step', alpha=.3,
                     common_norm=True, linestyle='-',
                     legend=True if i == 0 else False)

        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        # axes[i].set_xlim((50, 105))
        # axes[i].set_yticks(list())

        # axes[i].set_title(ID)

    plt.tight_layout()
    save_path = ROOT_DIR / 'results' / 'plots' / 'heartrate_hist.png'
    plt.savefig(save_path, dpi=600)
    plt.show()


def plot_heartrate_over_time(df: pd.DataFrame, feature: str = 'heartrate') -> None:
    f = plt.figure(figsize=(7.5, 10))
    axes = [f.add_subplot(8, 2, i + 1) for i in range(len(list(df['ID'].unique())))]

    for i, ID in enumerate(list(df['ID'].unique())):
        df_ = df.loc[df['ID'] == ID]
        df_ = df_.loc[df_['label'] == 'Fixation']

        if DEP_VAR_BINARY != 'label_hr_medsplit':

            if DEP_VAR_BINARY == 'label_hr':
                mean_hr = np.mean(df_[feature])
            elif DEP_VAR_BINARY == 'label_hr_median':
                mean_hr = np.median(df_[feature])
            else:
                raise UserWarning(f'Cannot plot heart rate split lines with feature {DEP_VAR_BINARY}')

            sd_hr = np.std(df_[feature])
            top = mean_hr + (sd_hr * SD_DEV_THRESH)
            bottom = mean_hr - (sd_hr * SD_DEV_THRESH)

        else:
            top = np.median(df_[feature])
            bottom = top

        sns.lineplot(x='chunk', y=feature, data=df_, ax=axes[i],
                     sort=False)
        axes[i].axhline(y=top, xmin=0, xmax=240, color='red', linestyle='--')
        axes[i].axhline(y=bottom, xmin=0, xmax=240, color='red', linestyle='--')

        # Set feature label only on first column
        if i % 2 == 0:
            axes[i].set_ylabel(feature)
        else:
            axes[i].set_ylabel('')

        axes[i].set_xlabel('')

    plt.tight_layout()
    save_path = ROOT_DIR / 'results' / 'plots' / f'{feature}_over_time.png'
    plt.savefig(save_path, dpi=600)
    plt.show()


def test_if_significant_from_mean(values: pd.Series, overall_mean: np.array) -> bool:
    alternative = 'less' if np.mean(values) < overall_mean else 'greater'
    t, p = ttest_1samp(values, overall_mean, alternative=alternative)

    if p < .05:
        return True
    else:
        return False


def plot_feature_importance(feature_explosion: bool, feature_reduction: bool, ) -> None:
    path = ROOT_DIR / 'results' / f'best_estimator_importances_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.csv'
    df = pd.read_csv(path)
    df = df.drop(['Unnamed: 0'], axis=1)

    features_ = [c.split() for c in df.columns]
    features = [f'{f[1]} {rename_features(f[0])}' for f in features_]
    df.columns = features

    # Sort dataframe by median
    df_sorted = df.reindex(df.mean().sort_values().index, axis=1)

    # Melt the dataframe so that feature and value get their own columns
    df_ = df_sorted.melt(var_name='Feature', value_name='Feature importance')
    df_['Movement type'] = df_['Feature'].apply(lambda x: x.split()[0])
    df_['Movement type'] = pd.Categorical(df_['Movement type'])

    plt.figure(figsize=(7.5, .33 * (len(list(df_['Feature'].unique())))))
    sns.barplot(y='Feature', x='Feature importance', data=df_,
                color='gray',
                capsize=.5, errwidth=1.2,
                orient='h')
    plt.axvline(x=np.mean(df_['Feature importance']), linestyle='--', color='red')
    plt.xlabel(f'Coefficient (explosion={feature_explosion}, reduction={feature_reduction})')

    # Compute the mean impurity for each feature, so we can use it later to determine the max and plot a * besides it
    feature_means = [np.mean(df_.loc[df_['Feature'] == feat]['Feature importance']) for feat in list(df_['Feature'].unique())]

    # Run a one_sample t-test for each feature, comparing it to the overall mean
    for i, feat in enumerate(list(df_['Feature'].unique())):
        df_feat = df_.loc[df_['Feature'] == feat]
        p = test_if_significant_from_mean(df_feat['Feature importance'],
                                          np.mean(df_['Feature importance']))

        if p:
            plt.text(x=max(feature_means) * 1.08, y=i, s='*',
                     color='red', ha='center', va='center',
                     fontsize=13)

    plt.tight_layout()
    savepath = ROOT_DIR / 'results' / 'plots' / f'feature_importances_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.png'
    plt.savefig(savepath, dpi=600)
    plt.show()


def compute_subplot_layout(columns: int) -> Tuple[int, int]:
    if columns % 4 == 0:
        return int(columns / 4), 4
    elif columns % 3 == 0:
        return int(columns / 3), 3
    else:
        return int(columns / 2), 2


def plot_linear_predictions(feature_explosion: bool, feature_reduction: bool, poly_deg: int) -> None:
    model, X, y, column_names, r2 = pickle.load(
        open(ROOT_DIR / 'results' / f'linear_estimator_POLYDEG_{poly_deg}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.p', 'rb')
    )

    if model is None:
        raise Warning('There was no best regression model.')

    y_pred = model.predict(X)

    palette = sns.color_palette()

    f = plt.figure(figsize=(7.5, 10))
    # rows, cols = compute_subplot_layout(len(column_names))
    # axes = [f.add_subplot(rows, cols, i + 1) for i in range(len(column_names))]

    # for i, col in enumerate(column_names):
    rows, cols = compute_subplot_layout(X.shape[1])
    axes = [f.add_subplot(rows, cols, i + 1) for i in range(X.shape[1])]

    for i in range(X.shape[1]):
        x = X[:, i]
        x_lin = np.linspace(min(x), max(x), num=100)
        coef = list(model.coef_)[i]
        intc = model.intercept_
        y_pred = [v * coef + intc for v in x_lin]
        # sns.lineplot(x=x, y=y, ax=axes[i], color=palette[0])
        sns.lineplot(x=x_lin, y=y_pred, ax=axes[i], color=palette[0], zorder=10)
        # sns.scatterplot(x=x, y=y, ax=axes[i], color=palette[1])

        # axes[i].set_xlabel(col)

        if i % cols == 0:
            axes[i].set_ylabel('Predicted heart rate')
        else:
            axes[i].set_ylabel('')

        axes[i].set_ylim((50, 100))

    f.tight_layout()
    f.show()


def plot_linear_predictions_scatter(feature_explosion: bool, feature_reduction: bool, poly_deg) -> None:
    model, X, y, _, r2 = pickle.load(
        open(ROOT_DIR / 'results' / f'linear_estimator_POLYDEG_{poly_deg}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.p', 'rb')
    )

    if model is None:
        raise Warning('There was no best regression model.')

    y_pred = model.predict(X)

    plt.figure()
    sns.scatterplot(x=y, y=y_pred, label=f'R2 = {r2}')

    plt.xlabel('True heart rate')
    plt.ylabel('Predicted heart rate')

    limits = (50, 100)
    plt.xlim(limits)
    plt.ylim(limits)

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / f'linear_estimator_POLYDEG_{poly_deg}_EXP{int(feature_explosion)}_RED{int(feature_reduction)}.png',
                dpi=600)
    plt.show()
