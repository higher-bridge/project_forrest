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
from typing import Any, Dict, List

import heartpy as hp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from constants import CHUNK_SIZE, HZ, HZ_HEART, N_JOBS, SD_DEV_THRESH
from utils.hessels_classifier import classify_hessels2018

# import remodnav
# from I2MC import I2MC



def run_hessels_classifier(files_et: List[Path], verbose: bool = True) -> None:
    if N_JOBS is None or N_JOBS == 1:  # Run single core
        # Loop through list of Paths
        for i, f in enumerate(files_et):
            print(f'Classifying dataset {i} of {len(files_et)}')
            events_df = classify_hessels2018(f, verbose=True)
            events_df.to_csv(str(f).replace('.tsv', '-extracted.tsv'), delimiter='\t')

    else:
        results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=verbose)(
            delayed(classify_hessels2018)(f) for f in files_et)

        for f, events_df in zip(files_et, results):
            events_df.to_csv(str(f).replace('.tsv', '-extracted.tsv'), delimiter='\t')


def detect_heartrate(signal: List[Any]) -> List[float]:
    working_data, measures = hp.process_segmentwise(signal, sample_rate=HZ_HEART,
                                                    segment_width=CHUNK_SIZE, segment_min_size=CHUNK_SIZE * 0.5,
                                                    segment_overlap=0.0, mode='fast')

    return measures['bpm']


def split_into_chunks(df: pd.DataFrame, measurement_type: str) -> pd.DataFrame:
    if measurement_type == 'heartrate':
        sampling_rate = HZ_HEART
    elif measurement_type == 'eyetracking':
        sampling_rate = HZ
    elif measurement_type == 'normdiff':
        sampling_rate = 250
    else:
        raise ValueError(f'split_into_chunks(): provide one of [heartrate, eyetracking, normdiff] as measurement_type.')

    # sampling_rate = HZ_HEART if measurement_type == 'heartrate' else HZ

    if measurement_type == 'heartrate':
        timestamps = np.arange(0, len(df) / sampling_rate, 1 / sampling_rate)
    elif measurement_type == 'eyetracking':
        timestamps = [onset + duration for onset, duration in zip(list(df['onset']), list(df['duration']))]
    else:
        timestamps = list(df['onset'])

    chunks = []
    current_chunk = 1
    max_time = CHUNK_SIZE

    for stamp in list(timestamps):
        if stamp >= max_time:
            current_chunk += 1
            max_time += CHUNK_SIZE

        chunks.append(current_chunk)

    df['chunk'] = chunks

    return df


def get_bpm_dict(df_hr: List[pd.DataFrame], ID_hr: List[str], verbose: bool = True) -> Dict:
    bpm_dict = {key: dict() for key in ID_hr}

    for df, ID in zip(df_hr, ID_hr):
        signal = list(df.loc[:, 1])
        bpm_values = detect_heartrate(signal)

        for chunk in range(len(bpm_values)):
            bpm_dict[ID][chunk] = bpm_values[chunk]

        if verbose:
            print(f'Mean of mean = {round(float(np.mean(bpm_values)), 2)} bpm over {len(bpm_values)} chunks.')

    return bpm_dict


def get_hr_labels(df: pd.DataFrame, center: str = 'mean') -> List[int]:
    if center == 'log':
        # Compute the logarithm of the heart rate values (deep copy so we don't change values in-place)
        df_log = df.copy(deep=True)
        df_log['heartrate'] = np.log(df_log['heartrate'])
        df_ = df_log.loc[df_log['label'] == 'FIXA']
    else:
        df_ = df.loc[df['label'] == 'FIXA']

    # Heart rate value is already the same value throughout the chunk.
    # Here we group it so that we can get one row per chunk, no actual values are changed
    df_avg = df_.groupby(['chunk']).agg('mean').reset_index()

    if center == 'mean' or center == 'log':
        distribution_center = np.nanmean(df_avg['heartrate'])
    elif center == 'median':
        distribution_center = np.nanmedian(df_avg['heartrate'])
    else:
        raise UserWarning('Specify either mean, median or log')

    sd_overall = np.std(df_avg['heartrate'])

    hr_label = []

    for hr in list(df['heartrate']):
        # Compute if heartrate is higher or lower than the overall mean +/- a number of standard deviations
        if hr > distribution_center + (sd_overall * SD_DEV_THRESH):
            hr_label.append(1)  # High
        elif hr < distribution_center - (sd_overall * SD_DEV_THRESH):
            hr_label.append(-1)  # Low
        else:
            hr_label.append(0)  # In between / normal

    return hr_label


def get_heartrate_metrics(df: pd.DataFrame, ID: str) -> pd.DataFrame:
    df['label_hr'] = get_hr_labels(df, 'mean')
    df['label_hr_median'] = get_hr_labels(df, 'median')
    df['label_hr_log'] = get_hr_labels(df, 'log')

    return df


def add_bpm_to_eyetracking(dfs: List[pd.DataFrame], IDs: List[str], bpm_dict: Dict):
    new_dfs = []

    for df, ID in zip(dfs, IDs):
        bpm_ID = bpm_dict[ID]
        df['heartrate'] = [0] * len(df)

        for chunk in list(bpm_ID.keys()):
            avg_hr = bpm_ID[chunk]
            # chunk_idx = np.argwhere(df['chunk'] == chunk)
            df.loc[df['chunk'] == chunk, 'heartrate'] = avg_hr

        df = get_heartrate_metrics(df, ID)
        new_dfs.append(df)

    return new_dfs

# def run_remodnav(files_et: List[Path], verbose: bool = True) -> None:
#     # REMoDNaV returns nothing but writes immediately to file
#     if N_JOBS is None or N_JOBS == 1:  # Run single core
#         results = []
#
#         for f in files_et:
#             fixations = remodnav.main([None,
#                                        str(f),
#                                        str(f).replace('.tsv', '-extracted.tsv'),
#                                        str(PX2DEG),
#                                        str(HZ),
#                                        '--min-fixation-duration', '0.06',
#                                        '--pursuit-velthresh', '2.0'])
#             results.append(fixations)
#
#     else:  # Run with parallelism
#         arg_list = []
#         for f in files_et:
#             in_file = str(f)
#             out_file = str(f).replace('.tsv', '-extracted.tsv')
#
#             arg_list.append([None,
#                              in_file,
#                              out_file,
#                              str(PX2DEG),
#                              str(HZ),
#                              '--min-fixation-duration', '0.06',
#                              '--pursuit-velthresh', '2.0'])
#
#         results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=verbose)(
#             delayed(remodnav.main)(args) for args in arg_list)


# def run_i2mc(files_et: List[Path]) -> None:
#     options = dict()
#     options['xres'] = 1280
#     options['yres'] = 546  # 1024 full screen
#     options['missingx'] = np.nan
#     options['missingy'] = np.nan
#     options['freq'] = 1000
#     options['disttoscreen'] = 63
#
#     # For the full screen:
#     # options['scrSz'] = (26.5, 21.2)  # 1280px / 1024px = 1.25 (5:4 ratio). 26.5 cm width -> 33.9 diag / 21.2 height
#     # Only the actual stimulus (video content), to which gaze coordinates were mapped:
#     options['scrSz'] = (26.5, 11.3)   # 1280px / 546px = 2.35 (47:20 ratio). 26.5 cm width -> 28.8 diag / 11.3 height
#
#     options['minFixDur'] = 60.0
#
#     # Loop through list of Paths
#     for f in files_et:
#         df = pd.read_csv(f, delimiter='\t', header=None)
#
#         # Add colnames, drop last two
#         df.columns = ['L_X', 'L_Y', 'pupilsize', 'frameno']
#         df = df.drop(['pupilsize', 'frameno'], axis=1)
#
#         # Data is steady 1kHz, but has no timestamps, so add (in ms)
#         df['time'] = np.arange(len(df))
#
#         # Run I2MC fixation detection
#         events, _, _ = I2MC(df, options)
#
#         # Save to df
#         events_df = pd.DataFrame(events)
#         events_df.to_csv(str(f).replace('.tsv', '-extracted.tsv'), delimiter='\t')
