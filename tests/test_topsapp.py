from hyp3_isce2.topsapp import TopsappBurstConfig


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
