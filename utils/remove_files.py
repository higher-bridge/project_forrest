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

import os
from pathlib import Path
from constants import ROOT_DIR


def remove_temp_files(directory, key='._'):
    """ Removes temp data files, marked with the ._ prefix which are created
    sometimes (at least by macOS) """

    files_removed = 0
    all_files = directory.glob(f'*{key}')
    
    for filename in all_files:
        os.remove(filename)
        files_removed += 1
            
    print(f'Removed {files_removed} files.')


if __name__ == '__main__':
    path = ROOT_DIR / 'data'

    remove_temp_files(path)
