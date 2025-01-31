from hyp3_isce2 import s1_auxcal


def test_download_aux_cal(tmp_path):
    """This test is slow because it download ~3MB of data.
    Might want to consider skipping unless doing integration testing.
    """
    aux_cal_dir = tmp_path / 'aux_cal'
    s1_auxcal.download_aux_cal(aux_cal_dir)

    assert (aux_cal_dir / 'S1A_AUX_CAL_V20150722T120000_G20151125T104733.SAFE/data/s1a-aux-cal.xml').exists()
    assert (aux_cal_dir / 'S1A_AUX_CAL_V20150722T120000_G20151125T104733.SAFE/manifest.safe').exists()
    assert (aux_cal_dir / 'S1A_AUX_CAL_V20150722T120000_G20151125T104733.SAFE/support/s1-object-types.xsd').exists()
    assert (aux_cal_dir / 'S1A_AUX_CAL_V20150722T120000_G20151125T104733.SAFE/support/s1-aux-cal.xsd').exists()

    assert (aux_cal_dir / 'S1B_AUX_CAL_V20160422T000000_G20160701T144618.SAFE/data/s1b-aux-cal.xml').exists()
    assert (aux_cal_dir / 'S1B_AUX_CAL_V20160422T000000_G20160701T144618.SAFE/manifest.safe').exists()
    assert (aux_cal_dir / 'S1B_AUX_CAL_V20160422T000000_G20160701T144618.SAFE/support/s1-object-types.xsd').exists()
    assert (aux_cal_dir / 'S1B_AUX_CAL_V20160422T000000_G20160701T144618.SAFE/support/s1-aux-cal.xsd').exists()
