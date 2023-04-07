import os

import pytest

from hyp3_isce2.topsapp import TopsappBurstConfig, run_topsapp_burst


def test_topsapp_burst_config(tmp_path):
    config = TopsappBurstConfig(
        reference_safe='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE',
        secondary_safe='S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11.SAFE',
        orbit_directory='orbits',
        aux_cal_directory='aux_cal',
        region_of_interest=[-118.0, 37.0, -117.0, 38.0],
        dem_filename='dem.tif',
        swath=1,
    )
    template = config.generate_template()
    assert 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE' in template
    assert 'S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11.SAFE' in template
    assert 'orbits' in template
    assert 'aux_cal' in template
    assert 'dem.tif' in template
    assert '[-118.0, 37.0, -117.0, 38.0]' in template
    assert '[1]' in template

    template_path = tmp_path / 'topsapp.xml'
    config.write_template(template_path)
    assert template_path.exists()


def test_run_topsapp_burst(tmp_path):
    with pytest.raises(IOError):
        run_topsapp_burst('topsApp.xml')

    config = TopsappBurstConfig(
        reference_safe='',
        secondary_safe='',
        orbit_directory='',
        aux_cal_directory='',
        region_of_interest=[0, 1, 2, 3],
        dem_filename='',
        swath=1,
        azimuth_looks=1,
        range_looks=1,
    )
    template_path = config.write_template(tmp_path / 'topsApp.xml')

    with pytest.raises(ValueError):
        run_topsapp_burst('notastep', config_xml=template_path)

    with pytest.raises(ValueError):
        run_topsapp_burst('preprocess', 'startup', config_xml=template_path)

    os.chdir(tmp_path)
    run_topsapp_burst('preprocess', config_xml=template_path)