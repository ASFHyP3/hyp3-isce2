from hyp3_isce2 import s1_auxcal


def test_download_aux_cal(tmp_path):
    """This test is slow because it download ~3MB of data.
    Might want to consider skipping unless doing integration testing.
    """

    aux_cal_dir = tmp_path / 'aux_cal'
    s1_auxcal.download_aux_cal(aux_cal_dir)
    assert (aux_cal_dir / 'S1A_AUX_CAL_V20190228T092500_G20210104T141310.SAFE').exists()
    assert (aux_cal_dir / 'S1B_AUX_CAL_V20190514T090000_G20210104T140612.SAFE').exists()
