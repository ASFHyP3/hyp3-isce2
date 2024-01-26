"""Tests for hyp3_isce2.merge_tops_bursts module, use single quotes"""
from datetime import datetime
from pathlib import Path

import hyp3_isce2.burst as burst_utils
import hyp3_isce2.merge_tops_bursts as merge


def test_to_burst_params():
    product = merge.BurstProduct(
        'granule',
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        666666,
        'IW1',
        'VV',
        0,
        Path('granule'),
        100,
        100,
        10,
        2,
        10,
        50,
        20,
        40,
        0.1,
        1,
        datetime(2020, 1, 1, 0, 0, 1),
        datetime(2020, 1, 2, 0, 0, 1),
        99,
    )
    assert product.to_burst_params() == burst_utils.BurstParams('granule', 'IW1', 'VV', 0)


def test_tmp(test_merge_dir):
    pass
