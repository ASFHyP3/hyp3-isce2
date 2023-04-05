import tempfile
from pathlib import Path

import pytest

from hyp3_isce2 import s1_auxcal


@pytest.fixture()
def tempdir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


def test_download_aux_cal(tempdir):
    aux_cal_dir = Path(tempdir) / 'aux_cal'
    s1_auxcal.download_aux_cal(aux_cal_dir)
    assert (aux_cal_dir / 'S1A_AUX_CAL_V20190228T092500_G20210104T141310.SAFE').exists()
    assert (aux_cal_dir / 'S1B_AUX_CAL_V20190514T090000_G20210104T140612.SAFE').exists()
