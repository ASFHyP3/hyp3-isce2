from re import match
from unittest.mock import patch

from hyp3_isce2 import packaging


def test_get_product_name():
    reference_name = 'S1_136231_IW2_20200604T022312_VV_7C85-BURST'
    secondary_name = 'S1_136231_IW2_20200616T022313_VV_5D11-BURST'

    name_20m = packaging.get_product_name(reference_name, secondary_name, pixel_spacing=20, slc=False)
    name_80m = packaging.get_product_name(reference_name, secondary_name, pixel_spacing=80, slc=False)

    assert match('[A-F0-9]{4}', name_20m[-4:]) is not None
    assert match('[A-F0-9]{4}', name_80m[-4:]) is not None

    assert name_20m.startswith('S1_136231_IW2_20200604_20200616_VV_INT20')
    assert name_80m.startswith('S1_136231_IW2_20200604_20200616_VV_INT80')


def test_get_product_name_multi_burst(test_data_dir):
    with patch.object(packaging, 'token_hex') as patched_token_hex:
        patched_token_hex.return_value = 'ab12'

        reference_safe = str(test_data_dir / 'packaging/multi_burst/S1A_IW_SLC__1SSV_20200604T022307_20200604T022318_032861_03CE65_C158.SAFE')
        secondary_safe = str(test_data_dir / 'packaging/multi_burst/S1A_IW_SLC__1SSV_20200616T022308_20200616T022319_033036_03D3A3_7466.SAFE')

        product_name = packaging.get_product_name(reference_safe, secondary_safe, pixel_spacing=20)
        assert product_name == 'S1_064-000000s1n00-136229s2n05-136229s3n04_IW_20200604_20200616_VV_INT20_AB12'

        product_name = packaging.get_product_name(reference_safe, secondary_safe, pixel_spacing=80)
        assert product_name == 'S1_064-000000s1n00-136229s2n05-136229s3n04_IW_20200604_20200616_VV_INT80_AB12'

        reference_safe = str(test_data_dir / 'packaging/multi_burst/S1A_IW_SLC__1SSV_20220814T125820_20220814T125829_044549_055128_F814.SAFE')
        secondary_safe = str(test_data_dir / 'packaging/multi_burst/S1A_IW_SLC__1SSV_20220907T125822_20220907T125830_044899_055CF1_B95E.SAFE')

        product_name = packaging.get_product_name(reference_safe, secondary_safe, pixel_spacing=40)
        assert product_name == 'S1_027-000000s1n00-056069s2n04-000000s3n00_IW_20220814_20220907_VV_INT40_AB12'


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
        multilook_position=None,
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
        multilook_position=None,
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
