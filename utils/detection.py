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
from scipy.stats import linregress
from joblib import Parallel, delayed

from constants import CHUNK_SIZE, HZ, HZ_HEART, N_JOBS, SD_DEV_THRESH, DETREND_HR
from utils.hessels_classifier import classify_hessels2020


def run_hessels_classifier(files_et: List[Path], verbose=11) -> None:
    print(f'Using Hessels et al. (2020) slow/fast phase classifier on {len(files_et)} datasets.')
    if N_JOBS is None or N_JOBS == 1:  # Run single core
        # Loop through list of Paths
        for i, f in enumerate(files_et):
            print(f'Classifying dataset {i + 1} of {len(files_et)}')
            _ = classify_hessels2020(f, verbose=True)

    else:
        _ = Parallel(n_jobs=N_JOBS, backend='loky', verbose=verbose)(delayed(classify_hessels2020)(f) for f in files_et)


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
    else:
        raise ValueError(f'split_into_chunks(): provide one of [heartrate, eyetracking] as measurement_type.')

    if measurement_type == 'heartrate':
        timestamps = np.arange(0, len(df) / sampling_rate, 1 / sampling_rate)
    else:  # measurement_type == 'eyetracking':
        timestamps = [onset + duration for onset, duration in zip(list(df['onset']), list(df['duration']))]

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
    print('Computing heart rate over all chunks...')
    bpm_dict = {key: dict() for key in ID_hr}

    for df, ID in zip(df_hr, ID_hr):
        signal = list(df.loc[:, 1])
        bpm_values = detect_heartrate(signal)

        if DETREND_HR:
            slope, intercept, r, p, se = linregress(np.arange(len(bpm_values)), np.array(bpm_values))
            bpm_values = [b - slope * i for i, b in enumerate(bpm_values)]

        for chunk in range(len(bpm_values)):
            bpm_dict[ID][chunk] = bpm_values[chunk]

        if verbose:
            print(f'Mean of mean = {np.mean(bpm_values).round(2)} bpm over {len(bpm_values)} chunks.')

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
    elif center == 'mediansplit':
        distribution_center = np.nanmedian(df_avg['heartrate'])
        hr_label = list(df['heartrate'].apply(lambda x: 1 if x > distribution_center else -1))
        return hr_label
    else:
        raise UserWarning('Specify either mean, median or log')

    sd_overall = np.nanstd(df_avg['heartrate'])

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


def get_heartrate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df['label_hr'] = get_hr_labels(df, 'mean')
    df['label_hr_median'] = get_hr_labels(df, 'median')
    df['label_hr_log'] = get_hr_labels(df, 'log')
    df['label_hr_medsplit'] = get_hr_labels(df, 'mediansplit')

    return df


def add_bpm_to_eyetracking(dfs: List[pd.DataFrame], IDs: List[str], bpm_dict: Dict):
    new_dfs = []

    for df, ID in zip(dfs, IDs):
        bpm_ID = bpm_dict[ID]
        df['heartrate'] = [0] * len(df)

        for chunk in list(bpm_ID.keys()):
            avg_hr = bpm_ID[chunk]
            df.loc[df['chunk'] == chunk, 'heartrate'] = avg_hr

        df = get_heartrate_metrics(df)
        new_dfs.append(df)

    return new_dfs
