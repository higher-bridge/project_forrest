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

Python implementation of slow/fast phase classifier as in Roy S. Hessels, Andrea J. van Doorn, Jeroen S. Benjamins,
Gijs A. Holleman & Ignace T. C. Hooge (2020). Task-related gaze control in human crowd navigation.
Attention, Perception, & Psychophysics 82, pp. 2482–2501. doi: 10.3758/s13414-019-01952-9

Original Matlab implementation can be found at:
https://github.com/dcnieho/GlassesViewer/tree/master/user_functions/HesselsEtAl2020
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from constants import (HESSELS_LAMBDA, HESSELS_MAX_ITER, HESSELS_MINFIX,
                       HESSELS_THR, HESSELS_WINDOW_SIZE, PX2DEG)


### STATISTICS FUNCTIONS ###
def get_amplitudes(start_x: np.ndarray, start_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray) -> np.ndarray:
    amps = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return amps * PX2DEG


def get_starts_ends(x: np.ndarray, y: np.ndarray, imarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                               np.ndarray, np.ndarray]:
    start_x, end_x = [np.nan], [np.nan]
    start_y, end_y = [np.nan], [np.nan]
    for start, end in zip(imarks[:-1], imarks[1:]):
        start_x.append(x[start])
        end_x.append(x[end])
        start_y.append(y[start])
        end_y.append(y[end])

    return np.array(start_x), np.array(start_y), np.array(end_x), np.array(end_y)


def get_durations(smarks: np.ndarray) -> np.array:
    durations = np.full(len(smarks), np.nan, dtype=float)
    durations[1:] = smarks[1:] - smarks[:-1]
    return durations


def get_velocities(vel: np.ndarray, imarks: np.ndarray, stat: str) -> List[float]:
    velocities = [np.nan]
    for start, end in zip(imarks[:-1], imarks[1:]):
        vels = vel[start:end]

        if stat == 'peak':
            velocities.append(np.max(vels))
        elif stat == 'mean':
            velocities.append(np.mean(vels))
        elif stat == 'median':
            velocities.append(np.median(vels))
        else:
            velocities.append(np.nan)
            raise UserWarning(f'Cannot compute {stat} of velocities')

    return velocities


### CLASSIFIER IMPLEMENTATION ###
def detect_velocity(p: np.ndarray, t: np.ndarray) -> np.array:
    delta_time_1 = t[1:-1] - t[0:-2]
    delta_pos_1  = p[1:-1] - p[0:-2]
    delta_time_2 = t[2:] - t[1:-1]
    delta_pos_2  = p[2:] - p[1:-1]

    # Compute velocities
    vel = ((delta_pos_1 / delta_time_1) + (delta_pos_2 / delta_time_2)) / 2

    # Initialize array of nan's and fill all (except first and last value) with the computed velocities
    velocities = np.full(len(delta_pos_1) + 2, np.nan, dtype=float)
    velocities[1:-1] = vel

    return velocities


def detect_switches(qvel: np.ndarray) -> Tuple[np.array, np.array]:
    v = np.full(len(qvel) + 2, False, dtype=bool)
    v[1:-1] = qvel
    v = v.astype(int)

    v0 = v[:-1]
    v1 = v[1:]
    switches = v0 - v1

    # If False - True (0 - 1), switch_on, if True - False (1 - 0), switch_off
    switch_on = np.argwhere(switches == -1)
    switch_off = np.argwhere(switches == 1) - 1  # Subtract 1 from each element

    return switch_on, switch_off


def fmark(vel: np.ndarray, ts: np.ndarray, thr: np.ndarray) -> np.array:
    qvel = vel < thr

    # Get indices of starts and ends of fixation candidates
    switch_on, switch_off = detect_switches(qvel)
    time_on, time_off = ts[switch_on.ravel()], ts[switch_off.ravel()]

    # Get durations of candidates and find at which indices they are long enough. Then select only those.
    time_deltas = time_off - time_on
    qfix = np.argwhere(time_deltas > HESSELS_MINFIX)
    time_on = time_on[qfix].ravel()
    time_off = time_off[qfix].ravel()

    # Combine the two lists and sort the timestamps
    times_sorted = sorted(np.concatenate([time_on, time_off]))
    return times_sorted


def threshold(vel: np.ndarray) -> np.array:
    # Retrieve where vel is neither below threshold (and not nan), and get indices of those positions
    q_vel = vel < HESSELS_THR
    valid_idxs = np.argwhere(q_vel).ravel()

    mean_vel = np.mean(vel[valid_idxs])
    std_vel = np.std(vel[valid_idxs])

    counter = 0
    prev_thr = 0

    while True:
        thr2 = mean_vel + (HESSELS_LAMBDA * std_vel)
        valid_idxs = np.where(vel < thr2)

        if round(thr2) == round(prev_thr) or counter >= HESSELS_MAX_ITER:
            break

        mean_vel = np.mean(vel[valid_idxs])
        std_vel = np.std(vel[valid_idxs])
        prev_thr = thr2
        counter += 1

    thr2 = mean_vel + (HESSELS_LAMBDA * std_vel)
    return np.array([thr2] * len(vel))


def load_data(f: Path, delimiter: str, header) -> pd.DataFrame:
    df = pd.read_csv(f, delimiter=delimiter, header=header)

    # Add colnames, drop last two (hardcoded for the studyforrest dataset, needs changing with other datasets)
    df.columns = ['x', 'y', 'pupilsize', 'frameno']
    df = df.drop(['pupilsize', 'frameno'], axis=1)

    # Data is steady 1kHz, but has no timestamps, so add (in ms)
    df['time'] = np.arange(len(df))

    return df


def classify_hessels2018(f: Path, delimiter: str = '\t', header = None, verbose: bool = False) -> pd.DataFrame:
    """
    Implementation of Hessels et al., 2020 (see docstring at top of file)
    :param f: A pathlib Path to file (in this case .tsv)
    :param header: Whether the file has column headers (e.g. None or 0), see pandas documentation
    :param delimiter: The delimiter in the file. Use None for normal csv
    :param verbose: A bool indicating whether to print progress
    :return: A pd.DataFrame with column labels [label, onset, duration, amp, avg_vel, med_vel, peak_vel,
                                                start_x, end_x, start_y, end_y]
    """
    df = load_data(f, delimiter, header)
    x = np.array(df['x'])
    y = np.array(df['y'])
    ts = np.array(df['time'])

    # Retrieve euclidean velocity from each datapoint to the next
    vx = detect_velocity(x, ts)
    vy = detect_velocity(y, ts)
    vel = np.sqrt(vx ** 2 + vy ** 2)  # Same as matlab 'hypot', or euclidean distance
    vel_dva = vel * PX2DEG  # Multiply pixel differentials with some factor to get degrees visual angle

    window_size = int(HESSELS_WINDOW_SIZE)
    last_window_start_idx = len(ts) - (window_size + 1)

    thr, ninwin = np.zeros(len(ts)), np.zeros(len(ts))

    for i in range(last_window_start_idx):
        idxs = np.arange(i, i + window_size - 1)

        window_thr = threshold(vel_dva[idxs])
        thr[idxs] += window_thr
        ninwin[idxs] += 1

        if verbose and i % round(last_window_start_idx / 10) == 0:
            print(f'Processed {i} of {last_window_start_idx} threshold windows '
                  f'({round((i/last_window_start_idx) * 100, 1)}%)')

    thr = thr / ninwin

    emarks = fmark(vel_dva, ts, thr)
    imarks = [i for i, x in enumerate(ts) if x in emarks]
    assert len(emarks) == len(imarks)

    imarks = [i + 1 for i in imarks if i % 2 == 0]  # Add 1 to all even numbers
    smarks = ts[imarks]

    # Alternate slow and fast phases
    phase_types = []
    for i, _ in enumerate(smarks):
        if i % 2 == 1:
            phase_types.append('slow')
        else:
            phase_types.append('fast')

    # Check for too much missing data
    for i in range(len(imarks) - 1):
        idxs = np.arange(imarks[i] + 1, imarks[i + 1] - 2)
        nans = np.sum(np.isnan(x[idxs]))
        if nans >= (len(idxs) / 2):
            phase_types[i] = 'none'

    start_x, start_y, end_x, end_y = get_starts_ends(x, y, imarks)

    # Retrieve some extra information from the data
    results = {'label':     phase_types,
               'onset':     smarks,
               'duration':  get_durations(smarks),
               'amp':       get_amplitudes(start_x, start_y, end_x, end_y),
               'avg_vel':   get_velocities(vel, imarks, 'mean'),
               'med_vel':   get_velocities(vel, imarks, 'median'),
               'peak_vel':  get_velocities(vel, imarks, 'peak'),
               'start_x':   start_x,
               'start_y':   start_y,
               'end_x':     end_x,
               'end_y':     end_y}

    results = pd.DataFrame(results)

    if verbose:
        print(df.head())

    return results
