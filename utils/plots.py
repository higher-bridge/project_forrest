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

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from constants import IND_VARS, DEP_VAR, ROOT_DIR, SD_DEV_THRESH
from utils.pipeline_helper import rename_features


def plot_heartrate(signal, timestamps=None):
    if timestamps == None:
        timestamps = np.arange(signal)

    plt.figure()
    plt.plot(timestamps, signal)
    plt.ylabel('Pulse oximetry signal')
    plt.xlabel('Time')
    plt.show()


def plot_feature_hist(df):
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
            df_low = df_move_type.loc[df_move_type['label_hr'] == 'low']
            df_high = df_move_type.loc[df_move_type['label_hr'] == 'high']

            sns.kdeplot(x=feature, data=df_low, ax=axes[i],
                        label='Low', color=palette[0],
                        fill=True, linewidth=2.5,
                        common_norm=False, linestyle='-',
                        clip=(0.0, 350))
            sns.kdeplot(x=feature, data=df_high, ax=axes[i],
                        label='High', color=palette[1],
                        fill=True, linewidth=2.5,
                        common_norm=False, linestyle='--',
                        clip=(0.0, 350))

            # Remove y-ticks
            axes[i].set_yticks(list())

            # Set feature label only on first column
            if i % 2 == 0:
                axes[i].set_ylabel(rename_features(feature))
            else:
                axes[i].set_ylabel('')

            # Set movement type label only beneath last row
            if i >= (nrows * ncols) - ncols:
                axes[i].set_xlabel(f'{move_type}s')
            else:
                axes[i].set_xlabel('')

            # Set legend only in top-left panel
            if i == 0:
                axes[i].legend(fontsize=10)

            if i <= 1:
                print(f'{move_type}, {len(df_low)} low, {len(df_high)} high,',
                      f'{len(df_move_type) - len(df_low) - len(df_high)} in between. {len(df_move_type)} total.')

            i += 1

    plt.tight_layout()
    save_path = ROOT_DIR / 'results' / 'plots' / 'feature_hist.png'
    plt.savefig(save_path, dpi=600)
    plt.show()


def plot_heartrate_hist(df):
    palette = sns.color_palette('tab10')

    f = plt.figure(figsize=(7.5, 10))
    axes = [f.add_subplot(8, 2, i + 1) for i in range(len(list(df['ID'].unique())))]

    for i, ID in enumerate(list(df['ID'].unique())):
        df_ = df.loc[df['ID'] == ID]

        df_low = df_.loc[df_['label_hr'] == 'low']
        df_high = df_.loc[df_['label_hr'] == 'high']

        sns.histplot(x='heartrate', data=df_low, ax=axes[i],
                    label='Low', color=palette[0],
                    stat='count', alpha=.3, multiple='layer', discrete=False, element='step',
                    common_norm=True, linestyle='-')
        sns.histplot(x='heartrate', data=df_high, ax=axes[i],
                    label='High', color=palette[1],
                    stat='count', alpha=.3, multiple='layer', discrete=False, element='step',
                    common_norm=True, linestyle='--')
        sns.histplot(x='heartrate', data=df_, ax=axes[i],
                    label='All', color=palette[2],
                    stat='count', alpha=.3, multiple='layer', discrete=False, element='step',
                    common_norm=True, linestyle='-.')

        if i == 0:
            axes[i].legend()

        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        # axes[i].set_yticks(list())

        # axes[i].set_title(ID)

    plt.tight_layout()
    save_path = ROOT_DIR / 'results' / 'plots' / 'heartrate_hist.png'
    plt.savefig(save_path, dpi=600)
    plt.show()


def plot_heartrate_over_time(df):
    f = plt.figure(figsize=(7.5, 10))
    axes = [f.add_subplot(8, 2, i + 1) for i in range(len(list(df['ID'].unique())))]

    for i, ID in enumerate(list(df['ID'].unique())):
        df_ = df.loc[df['ID'] == ID]
        df_ = df_.loc[df_['label'] == 'Fixation']

        mean_hr = np.mean(df_['heartrate'])
        sd_hr = np.std(df_['heartrate'])
        top = mean_hr + (sd_hr * SD_DEV_THRESH)
        bottom = mean_hr - (sd_hr * SD_DEV_THRESH)

        sns.lineplot(x='chunk', y='heartrate', data=df_, ax=axes[i],
                     sort=False)
        axes[i].axhline(y=top, xmin=0, xmax=240, color='red', linestyle='--')
        axes[i].axhline(y=bottom, xmin=0, xmax=240, color='red', linestyle='--')

        # Set feature label only on first column
        if i % 2 == 0:
            axes[i].set_ylabel('heartrate')
        else:
            axes[i].set_ylabel('')

        axes[i].set_xlabel('')

    plt.tight_layout()
    save_path = ROOT_DIR / 'results' / 'plots' / 'heartrate_over_time.png'
    plt.savefig(save_path, dpi=600)
    plt.show()


