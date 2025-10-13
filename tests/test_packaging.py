from unittest.mock import patch

from hyp3_isce2 import packaging


def test_get_burst_date():
    assert packaging._get_burst_date('S1_056072_IW2_20220814T125829_VV_67BC-BURST') == '20220814'
    assert packaging._get_burst_date('S1_056072_IW2_20220907T125830_VV_97A5-BURST') == '20220907'


def test_get_data_year():
    assert (
        packaging._get_data_year(
            [
                'S1_056072_IW2_20220907T125830_VV_97A5-BURST',
                'S1_056071_IW2_20220907T125827_VV_97A5-BURST',
                'S1_056070_IW2_20220907T125824_VV_97A5-BURST',
            ]
        )
        == '2022'
    )
    assert (
        packaging._get_data_year(
            [
                'S1_056072_IW2_20220907T125830_VV_97A5-BURST',
                'S1_056071_IW2_20230907T125827_VV_97A5-BURST',
                'S1_056070_IW2_20220907T125824_VV_97A5-BURST',
            ]
        )
        == '2023'
    )


def test_get_product_name():
    with patch.object(packaging, 'token_hex') as mock_token_hex:
        mock_token_hex.return_value = 'ab12'
        result = packaging.get_product_name(
            reference_scenes=[
                'S1_056072_IW2_20220814T125829_VV_67BC-BURST',
                'S1_056071_IW2_20220814T125826_VV_67BC-BURST',
                'S1_056070_IW2_20220814T125823_VV_67BC-BURST',
                'S1_056069_IW2_20220814T125820_VV_67BC-BURST',
            ],
            secondary_scenes=[
                'S1_056072_IW2_20220907T125830_VV_97A5-BURST',
                'S1_056071_IW2_20220907T125827_VV_97A5-BURST',
                'S1_056070_IW2_20220907T125824_VV_97A5-BURST',
                'S1_056069_IW2_20220907T125822_VV_97A5-BURST',
            ],
            relative_orbit=64,
            pixel_spacing=20,
            polarization='VV',
        )
        assert result == 'S1_064-000000s1n00-056069s2n04-000000s3n00_IW_20220814_20220907_VV_INT20_AB12'

        mock_token_hex.return_value = 'cd34'
        result = packaging.get_product_name(
            reference_scenes=[
                'S1_136233_IW2_20200604T022318_VV_A53B-BURST',
                'S1_136232_IW3_20200604T022316_VV_A53B-BURST',
                'S1_136232_IW2_20200604T022315_VV_7C85-BURST',
                'S1_136231_IW3_20200604T022313_VV_7C85-BURST',
                'S1_136231_IW2_20200604T022312_VV_7C85-BURST',
                'S1_136230_IW3_20200604T022311_VV_7C85-BURST',
                'S1_136230_IW2_20200604T022310_VV_7C85-BURST',
                'S1_136229_IW3_20200604T022308_VV_7C85-BURST',
                'S1_136229_IW2_20200604T022307_VV_7C85-BURST',
            ],
            secondary_scenes=[
                'S1_136233_IW2_20200616T022319_VV_79C9-BURST',
                'S1_136232_IW3_20200616T022317_VV_79C9-BURST',
                'S1_136232_IW2_20200616T022316_VV_5D11-BURST',
                'S1_136231_IW3_20200616T022314_VV_5D11-BURST',
                'S1_136231_IW2_20200616T022313_VV_5D11-BURST',
                'S1_136230_IW3_20200616T022312_VV_5D11-BURST',
                'S1_136230_IW2_20200616T022311_VV_5D11-BURST',
                'S1_136229_IW3_20200616T022309_VV_5D11-BURST',
                'S1_136229_IW2_20200616T022308_VV_5D11-BURST',
            ],
            relative_orbit=27,
            pixel_spacing=40,
            polarization='HH',
        )
        assert result == 'S1_027-000000s1n00-136229s2n05-136229s3n04_IW_20200604_20200616_HH_INT40_CD34'


def test_get_relative_orbit(test_data_dir):
    safe_path = (
        test_data_dir / 'packaging' / 'slc' / 'S1A_IW_SLC__1SDV_20250406T022008_20250406T022035_058630_07421F_93A7.SAFE'
    )
    assert packaging.get_relative_orbit(safe_path) == 108


def test_get_pixel_size():
    assert packaging.get_pixel_size('20x4') == 80.0
    assert packaging.get_pixel_size('10x2') == 40.0
    assert packaging.get_pixel_size('5x1') == 20.0


def test_make_parameter_file(test_data_dir, tmp_path):
    parameter_file = tmp_path / 'slc_parameters.txt'
    data_dir = test_data_dir / 'packaging/slc/'
    packaging.make_parameter_file(
        out_path=parameter_file,
        reference_scenes=['S1A_IW_SLC__1SDV_20250406T022008_20250406T022035_058630_07421F_93A7'],
        secondary_scenes=['S1A_IW_SLC__1SDV_20250418T022008_20250418T022035_058805_074946_C7D4'],
        azimuth_looks=20,
        range_looks=4,
        apply_water_mask=False,
        reference_safe_path=data_dir / 'S1A_IW_SLC__1SDV_20250406T022008_20250406T022035_058630_07421F_93A7.SAFE',
        secondary_safe_path=data_dir / 'S1A_IW_SLC__1SDV_20250418T022008_20250418T022035_058805_074946_C7D4.SAFE',
        processing_path=data_dir,
        dem_name='GLO_30',
        dem_resolution=30,
    )
    assert parameter_file.exists()
    assert parameter_file.read_text() == '\n'.join(
        [
            'Reference Granule: S1A_IW_SLC__1SDV_20250406T022008_20250406T022035_058630_07421F_93A7',
            'Secondary Granule: S1A_IW_SLC__1SDV_20250418T022008_20250418T022035_058805_074946_C7D4',
            'Reference Pass Direction: ASCENDING',
            'Reference Orbit Number: 58630',
            'Secondary Pass Direction: ASCENDING',
            'Secondary Orbit Number: 58805',
            'Baseline: -129.02564735771583',
            'UTC time: 8409.44936',
            'Heading: -15.3974576912639',
            'Spacecraft height: 693000.0',
            'Earth radius at nadir: 6337286.638938101',
            'Slant range near: 800421.8030922529',
            'Slant range center: 826758.6675801671',
            'Slant range far: 853095.5320680811',
            'Range looks: 4',
            'Azimuth looks: 20',
            'INSAR phase filter: yes',
            'Phase filter parameter: 0.5',
            'Range bandpass filter: no',
            'Azimuth bandpass filter: no',
            'DEM source: GLO_30',
            'DEM resolution (m): 30',
            'Unwrapping type: snaphu_mcf',
            'Speckle filter: yes',
            'Water mask: no\n',
        ]
    )

    parameter_file = tmp_path / 'single_burst_parameters.txt'
    data_dir = test_data_dir / 'packaging/single_burst/'
    packaging.make_parameter_file(
        out_path=parameter_file,
        reference_scenes=['S1_372326_IW3_20180628T151555_VV_4673-BURST'],
        secondary_scenes=['S1_372326_IW3_20190705T151602_VV_D62F-BURST'],
        azimuth_looks=10,
        range_looks=2,
        apply_water_mask=True,
        reference_safe_path=data_dir / 'S1B_IW_SLC__1SSV_20180628T151555_20180628T151555_011575_015476_33AD.SAFE',
        secondary_safe_path=data_dir / 'S1B_IW_SLC__1SSV_20190705T151602_20190705T151602_017000_01FFC4_26ED.SAFE',
        processing_path=data_dir,
        dem_name='GLO_30',
        dem_resolution=30,
    )
    assert parameter_file.exists()
    assert parameter_file.read_text() == '\n'.join(
        [
            'Reference Granule: S1_372326_IW3_20180628T151555_VV_4673-BURST',
            'Secondary Granule: S1_372326_IW3_20190705T151602_VV_D62F-BURST',
            'Reference Pass Direction: DESCENDING',
            'Reference Orbit Number: 11575',
            'Secondary Pass Direction: DESCENDING',
            'Secondary Orbit Number: 17000',
            'Baseline: 7.636373874371554',
            'UTC time: 54955.501598',
            'Heading: -163.184113207136',
            'Spacecraft height: 693000.0',
            'Earth radius at nadir: 6337286.638938101',
            'Slant range near: 802590.1853264123',
            'Slant range center: 827292.8619908537',
            'Slant range far: 851995.5386552949',
            'Range looks: 2',
            'Azimuth looks: 10',
            'INSAR phase filter: yes',
            'Phase filter parameter: 0.5',
            'Range bandpass filter: no',
            'Azimuth bandpass filter: no',
            'DEM source: GLO_30',
            'DEM resolution (m): 30',
            'Unwrapping type: snaphu_mcf',
            'Speckle filter: yes',
            'Water mask: yes\n',
        ]
    )
