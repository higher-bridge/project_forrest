import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.file_management import load_merged_files
dataframes, IDs = load_merged_files('eyetracking', suffix='*-processed.tsv')

# from collections import Counter
# import seaborn as sns
# df = pd.concat(dataframes)
# high = df.loc[df['label_hr_median'] == 1]
# low = df.loc[df['label_hr_median'] == -1]
#
# h = Counter(high["label"])["LOSS"]
# l = Counter(low["label"])["LOSS"]
#
# print(f'Loss in high: {h / len(high)}')
# print(f'Loss in low: {l / len(low)}')
#
# high_loss = high.loc[high['label'] == 'LOSS']
# low_loss = low.loc[low['label'] == 'LOSS']
#
# print(f'Mean duration of loss in high = {np.median(high_loss["duration"])}')
# print(f'Mean duration of loss in low = {np.median(low_loss["duration"])}')
#
# plt.figure(figsize=(7.5, 5))
# sns.kdeplot(data=high_loss, x='duration', hue='label_hr_median')
# sns.kdeplot(data=low_loss, x='duration', hue='label_hr_median')
# plt.xlim((-10, 40))
# plt.show()

# Feature hist
# from utils.pipeline_helper import group_by_chunks
# dataframes_grouped = group_by_chunks(dataframes, feature_explosion=False, flatten=False)
#
# from utils.plots import plot_feature_hist
# df = pd.concat(dataframes_grouped)
# plot_feature_hist(df=df, dpi=600)

# Feature importance
# from utils.plots import plot_feature_importance
# plot_feature_importance(False, False, dpi=600)

# Heart rate example
# from utils.plots import plot_heartrate_over_time
# df = dataframes_grouped[0]
# plot_heartrate_over_time(df=df, feature='heartrate')


# Fixation detection examples
from pathlib import Path
from constants import ROOT_DIR

import seaborn as sns
from scipy.signal import savgol_filter


def euclidean(x, y):
    locs = [(x_, y_) for x_, y_ in zip(x, y)]
    vel = np.full(len(locs), np.nan, dtype=float)

    for i, (xy, xyd) in enumerate(zip(locs[:-1], locs[1:])):
        d = [(loc1 - loc2) ** 2 for loc1, loc2 in zip(xy, xyd)]  # Get squared x and y deltas
        dist = np.sqrt(sum(d))                                   # sqrt of sum of distances (bonus trick: works in n dimensions)
        vel[i + 1] = (round(dist, 3))

    return vel


ID = '02'
run = '1'

path = Path(ROOT_DIR / 'data' / 'eyetracking')

gaze = pd.read_csv(path / f'sub-{ID}_ses-movie_func_sub-{ID}_ses-movie_task-movie_run-{run}_recording-eyegaze_physio.tsv',
                   sep='\t', header=None)
fixs = pd.read_csv(path / f'sub-{ID}_ses-movie_func_sub-{ID}_ses-movie_task-movie_run-{run}_recording-eyegaze_physio-extracted.tsv',
                   sep='\t')

gaze.columns = ['x', 'y', 'pupilsize', 'frameno']
gaze['time'] = np.arange(len(gaze['x']))


s_range = np.arange(0, 500000, step=5000)
e_range = np.arange(5000, 505000, step=5000)

y_label = 'dispersion'

for start, end in zip(s_range, e_range):

    g = gaze.loc[gaze['time'] >= start]
    g = g.loc[g['time'] <= end]

    x = savgol_filter(g['x'], 31, 2, mode='nearest')
    g['x'] = x

    y = savgol_filter(g['y'], 31, 2, mode='nearest')
    g['y'] = y

    dispersion = euclidean(x, y)
    dispersion = savgol_filter(dispersion, 31, 2, mode='nearest')
    g['dispersion'] = dispersion

    f = fixs.loc[fixs['onset'] >= (start / 1000)]
    f = f.loc[f['onset'] <= (end / 1000)]

    plt.figure(figsize=(7.5 * 1.5, 5 * 1.5))
    sns.lineplot(data=g, x='time', y=y_label, color='black', linewidth=.5)

    for i in range(len(f)):
        row = f.iloc[i]
        onset = row['onset'] * 1000
        offset = onset + row['duration'] * 1000
        label = row['label']

        # plt.vlines(x=[onset, offset], ymin=np.min(g['x']), ymax=np.max(g['x']),
        #             color='gray')

        if label == 'FIXA':
            color = 'blue'
        elif label == 'SACC':
            color = 'red'
        elif label == 'BLINK':
            color = 'green'
        else:
            color ='gray'

        plt.fill_betweenx(y=[np.min(g[y_label]), np.max(g[y_label])],
                          x1=onset, x2=offset,
                          color=color,
                          alpha=.1)

        plt.text(x=np.mean([onset, offset]), y=np.min(g[y_label]),
                 ha='center', s=label, rotation=90, fontsize=8)

    plt.savefig(f'/Users/4137477/Desktop/revision_plots/{int(start)}.png')
    plt.show()
    plt.close()

print()
