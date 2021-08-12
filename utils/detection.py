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

import time
from typing import Any, Dict, List

import heartpy as hp
import numpy as np
import pandas as pd
import remodnav
from joblib import Parallel, delayed

from constants import (CHUNK_SIZE, HZ, HZ_HEART, N_JOBS_REMODNAV, PX2DEG,
                       SD_DEV_THRESH)


def run_remodnav(files_et: List[str], verbose: bool = True) -> List[Any]:
    if N_JOBS_REMODNAV == 0:  # Run single core
        results = []

        for f in files_et:
            fixations = remodnav.main([None, str(f), str(f).replace('-merged.tsv', '-extracted.tsv'),
                                       str(PX2DEG), str(HZ)])
            results.append(fixations)

    else:  # Run with parallelism
        arg_list = []
        for f in files_et:
            arg1 = str(f)
            arg2 = str(f).replace('-merged.tsv', '-extracted.tsv')

            arg_list.append([None, arg1, arg2, str(PX2DEG), str(HZ)])

        results = Parallel(n_jobs=N_JOBS_REMODNAV, backend='loky', verbose=verbose)(
            delayed(remodnav.main)(args) for args in arg_list)

    return results


def detect_heartrate(signal: List[Any]) -> List[float]:
    working_data, measures = hp.process_segmentwise(signal, sample_rate=HZ_HEART,
                                                    segment_width=CHUNK_SIZE, segment_min_size=CHUNK_SIZE * 0.5,
                                                    segment_overlap=0.0, mode='fast')

    return measures['bpm']


def split_into_chunks(df: pd.DataFrame, measurement_type: str) -> pd.DataFrame:
    sampling_rate = HZ_HEART if measurement_type == 'heartrate' else HZ

    if measurement_type == 'heartrate':
        timestamps = np.arange(0, len(df) / sampling_rate, 1 / sampling_rate)
    else:
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
    bpm_dict = {key: dict() for key in ID_hr}

    for df, ID in zip(df_hr, ID_hr):
        start_id = time.time()
        # df_ = split_into_chunks(df, 'heartrate')

        # signal = list(df_.loc[:, 1])
        signal = list(df.loc[:, 1])
        bpm_values = detect_heartrate(signal)

        for chunk in range(len(bpm_values)):
            bpm_dict[ID][chunk] = bpm_values[chunk]

        if verbose:
            print(f'ID {ID}: {round(time.time() - start_id, 2)} seconds.',
                  f'Mean of mean = {round(float(np.mean(bpm_values)), 2)} bpm over {len(bpm_values)} chunks.')

    return bpm_dict


def get_heartrate_metrics(df: pd.DataFrame, ID: str) -> pd.DataFrame:
    df_ = df.loc[df['label'] == 'FIXA']
    df_avg = df_.groupby(['chunk']).agg('mean').reset_index()

    mean_overall = np.mean(df_avg['heartrate'])
    sd_overall = np.std(df_avg['heartrate'])
    se_overall = sd_overall / np.sqrt(len(df_avg))

    print(f'{ID}: Mean overall = {round(mean_overall, 2)} (SD = {round(sd_overall, 2)}, SE = {round(se_overall, 2)})')

    hr_label = []

    for hr in list(df['heartrate']):
        # Compute if heartrate is higher or lower than the overall mean +/- a number of standard deviations
        if hr > mean_overall + (sd_overall * SD_DEV_THRESH):
            hr_label.append(1)  # High
        elif hr < mean_overall - (sd_overall * SD_DEV_THRESH):
            hr_label.append(-1)  # Low
        else:
            hr_label.append(0)  # In between / normal

    df['label_hr'] = hr_label

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
