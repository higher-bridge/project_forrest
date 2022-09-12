import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.file_management import load_merged_files
dataframes, IDs = load_merged_files('eyetracking', suffix='*-processed.tsv')

# Feature hist
from utils.pipeline_helper import group_by_chunks
dataframes_grouped = group_by_chunks(dataframes, feature_explosion=False, flatten=False)

from utils.plots import plot_feature_hist
df = pd.concat(dataframes_grouped)
plot_feature_hist(df=df, dpi=900)

# Feature importance
from utils.plots import plot_feature_importance
plot_feature_importance(False, False, dpi=900)

# Explained variances
# from pathlib import Path
# from constants import ROOT_DIR
# import json
#
# var_dir = Path(ROOT_DIR / 'results' / 'explained_variances')
# files_binary = list(var_dir.glob('*cont0*'))
#
# all_json = {'move_type': [],
#             'feature': [],
#             'explained_variance_ratio': []
#             }
#
# for f in files_binary:
#     with open(f, 'r') as file:
#         j = json.load(file)
#
#         for i in range(len(j['move_type'])):
#             all_json['move_type'].append(j['move_type'][i])
#             all_json['feature'].append(j['feature'][i])
#             all_json['explained_variance_ratio'].append(sum(j['explained_variance_ratio'][i]))
#
# df = pd.DataFrame(all_json)
# df_grouped = df.groupby(['move_type', 'feature']).agg(np.nanmean).reset_index()
# df_grouped.columns = ['Movement type', 'Feature', 'Mean explained variance']
#
# df_grouped.to_excel(ROOT_DIR / 'results' / 'explained_variance.xlsx')
