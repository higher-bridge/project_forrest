from utils.load_data import load_merged_files, load_and_concatenate_files, get_list_of_files
from constants import PX2DEG, HZ

import remodnav

from joblib import Parallel, delayed


def run_remodnav(files_et, n_jobs=4):
    if n_jobs == 0:  # Run single core
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

        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=True)(
            delayed(remodnav.main)(args) for args in arg_list)

    return results

if __name__ == '__main__':
    # load_and_concatenate_files('eyetracking')
    # load_and_concatenate_files('heartrate')

    # df_et, ID_et = load_merged_files('eyetracking')
    # df_hr, ID_hr = load_merged_files('heartrate')

    files_et = get_list_of_files('eyetracking', '*-merged.tsv')
    files_hr = get_list_of_files('heartrate', '*-merged.tsv')

    results = run_remodnav(files_et, n_jobs=4)

