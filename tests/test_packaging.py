from re import match

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


def test_get_pixel_size():
    assert packaging.get_pixel_size('20x4') == 80.0
    assert packaging.get_pixel_size('10x2') == 40.0
    assert packaging.get_pixel_size('5x1') == 20.0


def test_make_parameter_file(test_data_dir, tmp_path):
    parameter_file = tmp_path / 'parameters.txt'
    data_dir = test_data_dir / 'packaging'
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
