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

from functools import partial

import numpy as np
import scipy.stats


def _Range(a):
    try:
        a = a[np.logical_not(np.isnan(a))]
        if len(a) < 1:
            return np.nan
        a = np.array(a)
        return float(a.max() - a.min())
    except:
        return np.nan


def _tenthPerc(a):
    try:
        a = a[np.logical_not(np.isnan(a))]

        if len(a) < 1:
            return np.nan
        a = np.array(a)
        return float(scipy.stats.scoreatpercentile(a, 10))
    except:
        return np.nan


def _ninetythPerc(a):
    try:
        a = a[np.logical_not(np.isnan(a))]

        if len(a) < 1:
            return np.nan
        a = np.array(a)
        return float(scipy.stats.scoreatpercentile(a, 90))
    except:
        return np.nan


def _interQ_range(a):
    try:
        a = a[np.logical_not(np.isnan(a))]

        if len(a) < 1:
            return np.nan
        a = np.array(a)
        return float(np.subtract(*np.percentile(a, [75, 25])))
    except:
        return np.nan


def _mean_abs_deviation(a):
    try:
        a = a[np.logical_not(np.isnan(a))]

        if len(a) < 1:
            return np.nan
        a = np.array(a)
        return float(np.mean(np.abs(a - a.mean())))
    except:
        return np.nan


def _energy(a):
    try:
        a = a[np.logical_not(np.isnan(a))]

        if len(a) < 1:
            return np.nan
        a = np.array(a)
        return float(np.multiply(a, a).sum())
    except:
        return np.nan


def _rms(a):
    try:
        a = a[np.logical_not(np.isnan(a))]

        if len(a) < 1:
            return np.nan
        a = np.array(a)
        return float(np.sqrt(np.divide(np.multiply(a, a).sum(), a.size)))
    except:
        return np.nan


def _entropy(a):
    try:
        a = a[np.logical_not(np.isnan(a))]

        if len(a) < 1:
            return np.nan
        a = np.array(a)
        probs = np.divide(scipy.stats.itemfreq(a)[:, 1], a.size)
        return float(scipy.stats.entropy(probs, base=2))
    except:
        return np.nan


def _uniformity(a):
    try:
        a = a[np.logical_not(np.isnan(a))]
        if len(a) < 1:
            return np.nan
        a = np.array(a)

        probs = np.divide(scipy.stats.itemfreq(a)[:, 1], a.size)
        return float(np.multiply(probs, probs).sum())
    except:
        return np.nan


def _Mean(a):
    try:
        return float(np.nanmean(a))
    except:
        return np.nan


def _Variance(a):
    try:
        return float(np.nanvar(a))
    except:
        return np.nan


def _Skew(a):
    try:
        return float(scipy.stats.skew(a, nan_policy='omit'))
    except:
        return np.nan


def _Kurtosis(a):
    try:
        return float(scipy.stats.kurtosis(a, nan_policy='omit'))
    except:
        return np.nan


stat_features = [
    partial(_Mean),
    partial(_Variance),
    partial(_Skew),
    partial(_Kurtosis),
    partial(_Range),
    partial(_tenthPerc),
    partial(_ninetythPerc),
    partial(_interQ_range),
    partial(_mean_abs_deviation),
    partial(_energy),
    partial(_rms),
    partial(_entropy),
    partial(_uniformity)
]
