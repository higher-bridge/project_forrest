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

------------------------------
Python implementation of slow/fast phase classifier as in Roy S. Hessels, Andrea J. van Doorn, Jeroen S. Benjamins,
Gijs A. Holleman & Ignace T. C. Hooge (2020). Task-related gaze control in human crowd navigation.
Attention, Perception, & Psychophysics 82, pp. 2482–2501. doi: 10.3758/s13414-019-01952-9

Original Matlab implementation (and remaining documentation) can be found at:
https://github.com/dcnieho/GlassesViewer/tree/master/user_functions/HesselsEtAl2020
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from constants import (HESSELS_LAMBDA, HESSELS_MAX_ITER, HESSELS_MIN_FIX, HESSELS_THR, HESSELS_WINDOW_SIZE,
                       HESSELS_MIN_AMP, PX2DEG, PURSUIT_AS_FIX, PURSUIT_THR, HZ)


### STATISTICS FUNCTIONS ###
def _get_amplitudes(start_x: np.ndarray, start_y: np.ndarray,
                    end_x: np.ndarray, end_y: np.ndarray,
                    to_dva: bool = True) -> np.ndarray:
    amps = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    if to_dva:
        amps *= PX2DEG

    return amps.round(3)


def _get_starts_ends(x: np.ndarray, y: np.ndarray, imarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                                np.ndarray, np.ndarray]:
    start_x, end_x = [], []
    start_y, end_y = [], []
    for start, end in zip(imarks[:-1], imarks[1:]):
        start_x.append(x[start])
        end_x.append(x[end])
        start_y.append(y[start])
        end_y.append(y[end])

    start_x.append(np.nan)
    end_x.append(np.nan)
    start_y.append(np.nan)
    end_y.append(np.nan)

    return np.array(start_x).round(2), np.array(start_y).round(2), np.array(end_x).round(2), np.array(end_y).round(2)


def _get_durations(smarks: np.ndarray) -> np.ndarray:
    durations = np.full(len(smarks), np.nan, dtype=float)
    durations[:-1] = smarks[1:] - smarks[:-1]
    return durations.round(3)


def _get_velocities(vel: np.ndarray, imarks: np.ndarray, stat: str, to_dva: bool = True) -> np.ndarray:
    velocities = []
    for start, end in zip(imarks[:-1], imarks[1:]):
        vels = vel[start:end]

        if stat == 'peak':
            velocities.append(np.max(vels))
        elif stat == 'mean':
            velocities.append(np.nanmean(vels))
        elif stat == 'median':
            velocities.append(np.nanmedian(vels))
        else:
            velocities.append(np.nan)
            raise UserWarning(f'Cannot compute {stat} of velocities')

    velocities.append(np.nan)
    velocities = np.array(velocities)

    if to_dva:
        velocities *= PX2DEG

    velocities *= HZ

    return velocities.round(3)


def _split_slow_phase(df: pd.DataFrame) -> pd.DataFrame:
    vel = list(df['avg_vel'])
    label = list(df['label'])

    # Rename 'faster' slow phases (above PURSUIT_THR) to 'purs'
    for i, v in enumerate(vel):
        if label[i] == 'FIXA' and v > PURSUIT_THR:
            label[i] = 'PURS'

    df['label'] = label

    return df


### CLASSIFIER IMPLEMENTATION ###
def detect_velocity(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    I think this causes some rounding errors? Or at least gives different results than detect_velocity_python()
    :param p: array of position over time (x or y)
    :param t: array of timestamps
    :return:
    """
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


def detect_velocity_python(x: np.ndarray, y: np.ndarray, ts: np.ndarray) -> np.ndarray:
    # Assert that measurement rate is constant.
    delta_time = ts[1:] - ts[:-1]
    assert np.nanstd(delta_time) <= .1, 'Use detect_velocity() for non-constant measurement rates!'

    # Compile x/y arrays into tuples
    locs = [(x_, y_) for x_, y_ in zip(x, y)]

    # Create empty array with velocities
    vel = np.full(len(locs), np.nan, dtype=float)

    # Loop through locations + staggered locations (delta +1) and get the euclidean distance.
    # Since time delta is constant we don't need to take that into account
    for i, (xy, xyd) in enumerate(zip(locs[:-1], locs[1:])):
        d = [(loc1 - loc2) ** 2 for loc1, loc2 in zip(xy, xyd)]  # Get squared x and y deltas
        dist = np.sqrt(sum(d))                                   # sqrt of sum of distances (bonus trick: works in n dimensions)
        vel[i + 1] = (round(dist, 3))                            # Measurement rate is constant, so distance == velocity

    return vel


def detect_switches(qvel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.full(len(qvel) + 2, False, dtype=bool)
    v[1:-1] = qvel
    v = v.astype(int)

    v0 = v[:-1]
    v1 = v[1:]
    switches = v0 - v1

    # If False - True (0 - 1): switch_on, if True - False (1 - 0): switch_off
    switch_on = np.argwhere(switches == -1)
    switch_off = np.argwhere(switches == 1) - 1  # Subtract 1 from each element

    return switch_on.ravel(), switch_off.ravel()


def merge_fix_candidates(idxs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # idxs contains only starts, with slow/fast alternating
    fast_starts = idxs[1:-1:2]  # Take 1st element (0th is slow phase) and keep skipping two
    fast_ends = idxs[2::2] - 1  # Stagger by 1 so we get the start indices of the next slow phase, and remove 1 from all

    # Loop through start/end indices of the fast phases and determine if amplitude is too low
    remove_from_idxs = []
    for s, e in zip(fast_starts, fast_ends):
        amp = _get_amplitudes(x[s], y[s], x[e], y[e], to_dva=True)
        if amp < HESSELS_MIN_AMP:
            remove_from_idxs.append(s)
            remove_from_idxs.append(e + 1)  # We did ends - 1, so now we have to add it back to get a valid index

    mask = [i for i, x in enumerate(idxs) if x in remove_from_idxs]
    keep_idxs = np.delete(idxs, mask)

    return keep_idxs


def fmark(vel: np.ndarray, ts: np.ndarray, thr: np.ndarray) -> np.ndarray:
    qvel = vel < thr

    # Get indices of starts and ends of fixation candidates
    switch_on, switch_off = detect_switches(qvel)
    time_on, time_off = ts[switch_on], ts[switch_off]

    # Get durations of candidates and find at which indices they are long enough. Then select only those.
    time_deltas = time_off - time_on
    qfix = np.argwhere(time_deltas > HESSELS_MIN_FIX)
    time_on = time_on[qfix].ravel()
    time_off = time_off[qfix].ravel()

    # Combine the two lists and sort the timestamps
    times_sorted = sorted(np.concatenate([time_on, time_off]))

    return np.array(times_sorted)


def threshold(vel: np.ndarray) -> np.array:
    # Retrieve where vel is neither below threshold (and not nan), and get indices of those positions
    valid_idxs = np.argwhere(vel < HESSELS_THR).ravel()

    mean_vel = np.nanmean(vel[valid_idxs])
    std_vel = np.nanstd(vel[valid_idxs])

    if np.isnan(mean_vel):
        # print('Too much data loss. Could not compute velocity')
        return np.array([np.nan] * len(vel))

    counter = 0
    prev_thr = HESSELS_THR

    while True:
        thr2 = mean_vel + (HESSELS_LAMBDA * std_vel)
        valid_idxs = np.argwhere(vel < thr2).ravel()

        if round(thr2, 8) == round(prev_thr, 8) or counter >= HESSELS_MAX_ITER:
            break

        mean_vel = np.nanmean(vel[valid_idxs])
        std_vel = np.nanstd(vel[valid_idxs])
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


def classify_hessels2020(f: Path, delimiter='\t', header=None, verbose: bool = False) -> bool:
    """
    Implementation of Hessels et al., 2020 (see docstring at top of file)
    :param f: A pathlib Path to file (in this case .tsv)
    :param header: Whether the file has column headers (e.g. None or 0), see pandas documentation
    :param delimiter: The delimiter in the file. Use None for normal csv
    :param verbose: A bool indicating whether to print progress
    :return: A bool which indicates success or failure. Dataframe is immediately written to file
    """
    np.seterr(divide='ignore', invalid='ignore', )

    try:
        df = load_data(f, delimiter, header)
        x_before = np.array(df['x'])
        y_before = np.array(df['y'])
        ts = np.array(df['time']).astype(int)

        filter_len = 31
        x = savgol_filter(x_before, filter_len, 2, mode='nearest')
        y = savgol_filter(y_before, filter_len, 2, mode='nearest')

        # Retrieve euclidean velocity from each datapoint to the next
        # vx = detect_velocity(x, ts)
        # vy = detect_velocity(y, ts)
        # vel = np.sqrt(vx ** 2 + vy ** 2)

        vel = detect_velocity_python(x, y, ts)

        window_size = int(HESSELS_WINDOW_SIZE)
        last_window_start_idx = len(ts) - window_size  # (window_size + 1)

        thr, ninwin = np.zeros(len(ts)), np.zeros(len(ts))

        start_indices = list(np.arange(0, last_window_start_idx, step=100))
        for i in start_indices:
            idxs = np.arange(i, i + window_size + 1)

            window_thr = threshold(vel[idxs])
            thr[idxs] += window_thr
            ninwin[idxs] += 1

            if verbose and i % round(last_window_start_idx / 5) == 0:
                print(f'Processed {i} of {len(start_indices)} threshold windows '
                      f'({round((i / len(start_indices)) * 100)}%)', end='\r')

        thr /= ninwin

        emarks = fmark(vel, ts, thr)  # Get slow events

        # Get indices of timestamps if they are in emarks
        imarks = [i for i, t in enumerate(ts) if t in emarks]  # Get index of timestamp if that ts is found in emarks
        assert len(emarks) == len(imarks), 'Not all output samples have a corresponding input time!'

        starts, ends = np.array(imarks[::2]), np.array(imarks[1::2])
        imarks = np.array(sorted(np.concatenate([starts, ends + 1])))

        imarks = merge_fix_candidates(imarks, x, y)

        try:
            smarks = ts[imarks]
        except IndexError as e:
            print(f, e)
            imarks[-1] -= 1
            smarks = ts[imarks]

        # Alternate slow and fast phases
        phase_types = []
        for i, _ in enumerate(imarks):
            if i % 2 == 0:
                phase_types.append('FIXA')
            else:
                phase_types.append('SACC')

        # Check for too much missing data
        for i in range(len(imarks) - 1):
            idxs = np.arange(imarks[i] + 1, imarks[i + 1] - 2)
            nans = np.sum(np.isnan(x[idxs]))
            if nans >= (len(idxs) / 2):
                phase_types[i] = 'none'

        # Retrieve some extra information from the data
        start_x, start_y, end_x, end_y = _get_starts_ends(x, y, imarks)
        results = {'label':     phase_types,
                   'onset':     smarks / 1000,
                   'duration':  _get_durations(smarks) / 1000,
                   'amp':       _get_amplitudes(start_x, start_y, end_x, end_y),
                   'avg_vel':   _get_velocities(vel, imarks, 'mean'),
                   'med_vel':   _get_velocities(vel, imarks, 'median'),
                   'peak_vel':  _get_velocities(vel, imarks, 'peak'),
                   'start_x':   start_x,
                   'start_y':   start_y,
                   'end_x':     end_x,
                   'end_y':     end_y}

        results = pd.DataFrame(results)
        results = results.loc[results['duration'] >= .03]  # Remove all rows where duration < 30 ms
        results = results.loc[results['duration'] <= 5.0]
        results = results.loc[results['peak_vel'] <= 1000]

        if not PURSUIT_AS_FIX:
            results = _split_slow_phase(results)

        if verbose:
            print(results.head())

        results.to_csv(str(f).replace('.tsv', '-extracted.tsv'), sep=delimiter)
        return True

    except Exception as e:
        print(f'Error while classifying {str(f)}: {e}')

        results = {'label':     [np.nan],
                   'onset':     [np.nan],
                   'duration':  [np.nan],
                   'amp':       [np.nan],
                   'avg_vel':   [np.nan],
                   'med_vel':   [np.nan],
                   'peak_vel':  [np.nan],
                   'start_x':   [np.nan],
                   'start_y':   [np.nan],
                   'end_x':     [np.nan],
                   'end_y':     [np.nan]}
        results = pd.DataFrame(results)
        results.to_csv(str(f).replace('.tsv', '-extracted.tsv'), sep=delimiter)

        return False
