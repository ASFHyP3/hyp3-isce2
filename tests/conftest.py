import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def test_data_dir():
    here = Path(os.path.dirname(__file__))
    return here / 'data'


@pytest.fixture()
def test_merge_dir(test_data_dir):
    merge_dir = test_data_dir / 'merge'

    if not merge_dir.exists():
        print('Unzipping merge test data...')
        merge_zip = merge_dir.with_suffix('.zip')

        if not merge_zip.exists():
            raise ValueError('merge data not present, run data/create_merge_test_data.py')

        shutil.unpack_archive(merge_zip, test_data_dir)

    return merge_dir
