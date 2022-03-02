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

from src.main_pipeline import run_single_pipeline
from utils.pipeline_helper import print_overall_performance


### A quick script to handle all combinations of pipelines that we're interested in ###
def main():
    # Run just the plots
    run_single_pipeline(group=False, plot=True, process=False, preselection=False, search=False, regression=False)

    # Run three combinations of feature explosion/reduction (avoid explosion=False with reduction=True)
    feat_explosion = [False, True, True]
    feat_reduction = [False, False, True]

    for explosion, reduction in zip(feat_explosion, feat_reduction):
        # Run model preselection and search, and non-polynomial regression
        run_single_pipeline(group=False, plot=False, process=True, preselection=True, search=True, regression=True,
                            feature_explosion=explosion, feature_reduction=reduction, poly_degree=1)

        # Now run just the regression, this time polynomial with degree = 2 (poly degree does not influence
        # non-regression models, so this way we save a lot of time)
        run_single_pipeline(group=False, plot=False, process=True, preselection=False, search=False, regression=True,
                            feature_explosion=explosion, feature_reduction=reduction, poly_degree=2)

    print_overall_performance()


main()
