"""Tests for the single-burst specific functionality found in burst.py"""

from collections import namedtuple
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import asf_search
import numpy as np
import pytest
from lxml import etree
from shapely import geometry

from hyp3_isce2 import burst, utils


URL_BASE = 'https://datapool.asf.alaska.edu/SLC'
REF_DESC = burst.BurstParams(
    'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
    'IW2',
    'VV',
    3,
)
SEC_DESC = burst.BurstParams(
    'S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11',
    'IW2',
    'VV',
    3,
)
REF_ASC = burst.BurstParams(
    'S1A_IW_SLC__1SDV_20200608T142544_20200608T142610_032927_03D069_14F4',
    'IW1',
    'VV',
    1,
)
SEC_ASC = burst.BurstParams(
    'S1A_IW_SLC__1SDV_20200620T142544_20200620T142611_033102_03D5B7_8F1B',
    'IW1',
    'VV',
    1,
)


def load_metadata(metadata):
    metadata_path = Path(__file__).parent.absolute() / 'data' / metadata
    xml = etree.parse(metadata_path).getroot()
    return xml


def make_test_image(output_path, array=None):
    parent = Path(output_path).parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    if array is None:
        array = np.zeros((100, 100), dtype=np.float32)
        array[10:90, 10:90] = 1

    array = array.astype(np.float32)
    img_obj = utils.create_image(output_path, array.shape[1], access_mode='write', action='create')
    utils.write_isce2_image_from_obj(img_obj, array)


@pytest.mark.parametrize(
    'pattern',
    (
        '*SAFE',
        '*SAFE/annotation/*xml',
        '*SAFE/annotation/calibration/calibration*xml',
        '*SAFE/annotation/calibration/noise*xml',
        '*SAFE/measurement/*tiff',
    ),
)
def test_spoof_safe(tmp_path, mocker, pattern):
    mock_tiff = tmp_path / 'test.tiff'
    mock_tiff.touch()

    ref_burst = burst.BurstMetadata(load_metadata('reference_descending.xml'), REF_DESC)
    burst.spoof_safe(ref_burst, mock_tiff, tmp_path)
    assert len(list(tmp_path.glob(pattern))) == 1


@pytest.mark.parametrize('orbit', ('ascending', 'descending'))
def test_get_region_of_interest(tmp_path, orbit):
    """Test that the region of interest is correctly calculated for a given burst pair.
    Specifically, the region of interest we create should intersect the bursts used to create it,
    but not the bursts before or after it. This is difficult due to to the high degree of overlap between bursts.

    This diagram shows the burst layout for a descending orbit (the 0 indicates the region of interest):
          +---------------+
          |               |
       +--+------------+  |
       |  |            |  |
    +--+--+---------+--+--+
    |  |            |  |
    |  +------------+--0
    |               |
    +---------------+
    The diagram for an ascending orbit is the same, but rotated 180 degrees.
    """
    options = {'descending': [REF_DESC, SEC_DESC], 'ascending': [REF_ASC, SEC_ASC]}

    params = options[orbit]

    for param, name in zip(params, ('reference', 'secondary')):
        mock_tiff = tmp_path / 'test.tiff'
        mock_tiff.touch()
        burst_metadata = burst.BurstMetadata(load_metadata(f'{name}_{orbit}.xml'), param)
        burst.spoof_safe(burst_metadata, mock_tiff, tmp_path)

    sec_bbox = burst.get_isce2_burst_bbox(params[1], tmp_path)

    granule = params[0].granule
    burst_number = params[0].burst_number
    swath = params[0].swath
    pol = params[0].polarization
    ref_bbox_pre = burst.get_isce2_burst_bbox(burst.BurstParams(granule, swath, pol, burst_number - 1), tmp_path)
    ref_bbox_on = burst.get_isce2_burst_bbox(burst.BurstParams(granule, swath, pol, burst_number), tmp_path)
    ref_bbox_post = burst.get_isce2_burst_bbox(burst.BurstParams(granule, swath, pol, burst_number + 1), tmp_path)

    asc = orbit == 'ascending'
    roi = geometry.box(*burst.get_region_of_interest(ref_bbox_on, sec_bbox, asc))

    assert not roi.intersects(ref_bbox_pre)
    assert roi.intersects(ref_bbox_on)
    assert not roi.intersects(ref_bbox_post)


def mock_asf_search_results(
    slc_name: str, subswath: str, polarization: str, burst_index: int
) -> asf_search.ASFSearchResults:
    product = asf_search.ASFProduct()
    product.umm = {'InputGranules': [slc_name]}
    product.properties.update(
        {
            'burst': {'subswath': subswath, 'burstIndex': burst_index},
            'polarization': polarization,
        }
    )
    results = asf_search.ASFSearchResults([product])
    results.searchComplete = True
    return results


def test_get_burst_params_08F8():
    with patch.object(asf_search, 'search') as mock_search:
        mock_search.return_value = mock_asf_search_results(
            slc_name='S1A_IW_SLC__1SDV_20230526T190821_20230526T190847_048709_05DBA8_08F8-SLC',
            subswath='IW3',
            polarization='VV',
            burst_index=8,
        )
        assert burst.get_burst_params('S1_346041_IW3_20230526T190843_VV_08F8-BURST') == burst.BurstParams(
            granule='S1A_IW_SLC__1SDV_20230526T190821_20230526T190847_048709_05DBA8_08F8',
            swath='IW3',
            polarization='VV',
            burst_number=8,
        )
        mock_search.assert_called_once_with(product_list=['S1_346041_IW3_20230526T190843_VV_08F8-BURST'])


def test_get_burst_params_1B3B():
    with patch.object(asf_search, 'search') as mock_search:
        mock_search.return_value = mock_asf_search_results(
            slc_name='S1A_EW_SLC__1SDH_20230526T143200_20230526T143303_048706_05DB92_1B3B-SLC',
            subswath='EW5',
            polarization='HH',
            burst_index=19,
        )
        assert burst.get_burst_params('S1_308695_EW5_20230526T143259_HH_1B3B-BURST') == burst.BurstParams(
            granule='S1A_EW_SLC__1SDH_20230526T143200_20230526T143303_048706_05DB92_1B3B',
            swath='EW5',
            polarization='HH',
            burst_number=19,
        )
        mock_search.assert_called_with(product_list=['S1_308695_EW5_20230526T143259_HH_1B3B-BURST'])


def test_get_burst_params_burst_does_not_exist():
    with patch.object(asf_search, 'search') as mock_search:
        mock_search.return_value = []
        with pytest.raises(ValueError, match=r'.*failed to find.*'):
            burst.get_burst_params('this burst does not exist')
        mock_search.assert_called_once_with(product_list=['this burst does not exist'])


def test_get_burst_params_multiple_results():
    with patch.object(asf_search, 'search') as mock_search:
        mock_search.return_value = ['foo', 'bar']
        with pytest.raises(ValueError, match=r'.*found multiple results.*'):
            burst.get_burst_params('there are multiple copies of this burst')
        mock_search.assert_called_once_with(product_list=['there are multiple copies of this burst'])


def test_num_swath_pol():
    assert burst._num_swath_pol('S1_136231_IW2_20200604T022312_VV_7C85-BURST') == '136231_IW2_VV'
    assert burst._num_swath_pol('S1_068687_IW3_20230423T223824_HH_BA77-BURST') == '068687_IW3_HH'


def test_burst_datetime():
    assert burst._burst_datetime('S1_136231_IW2_20200604T022312_VV_7C85-BURST') == datetime(2020, 6, 4, 2, 23, 12)
    assert burst._burst_datetime('S1_068687_IW3_20230423T223824_HH_BA77-BURST') == datetime(2023, 4, 23, 22, 38, 24)
    assert burst._burst_datetime('S1_068687_IW3_20230403T020804_HH_BA77-BURST') == datetime(2023, 4, 3, 2, 8, 4)


def test_validate_bursts_list_length():
    burst.validate_bursts(
        'S1_000000_IW1_20200101T000000_VV_0000-BURST',
        'S1_000000_IW1_20200201T000000_VV_0000-BURST',
    )
    burst.validate_bursts(
        ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        ['S1_000000_IW1_20200201T000000_VV_0000-BURST'],
    )

    with pytest.raises(ValueError, match=r'^Must include at least 1 reference scene and 1 secondary scene$'):
        burst.validate_bursts(['a'], [])

    with pytest.raises(ValueError, match=r'^Must include at least 1 reference scene and 1 secondary scene$'):
        burst.validate_bursts([], ['a'])

    with pytest.raises(ValueError, match=r'^Must include at least 1 reference scene and 1 secondary scene$'):
        burst.validate_bursts([], [])

    with pytest.raises(
        ValueError,
        match=r'^Must provide the same number of reference and secondary scenes, got 2 reference and 1 secondary$',
    ):
        burst.validate_bursts(['a', 'b'], ['c'])


def test_validate_bursts_num_swath_pol():
    burst.validate_bursts(
        ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        ['S1_000000_IW1_20200201T000000_VV_0000-BURST'],
    )
    burst.validate_bursts(
        [
            'S1_000000_IW1_20200101T000000_VV_0000-BURST',
            'S1_000001_IW2_20200101T000001_VV_0000-BURST',
        ],
        [
            'S1_000000_IW1_20200201T000000_VV_0000-BURST',
            'S1_000001_IW2_20200201T000001_VV_0000-BURST',
        ],
    )

    with pytest.raises(
        ValueError,
        match=r'^Number \+ swath \+ polarization identifier does not match for reference scene S1_000001_IW2_20200101T000001_VV_0000\-BURST and secondary scene S1_000002_IW2_20200201T000001_VV_0000\-BURST$',
    ):
        # Different number
        burst.validate_bursts(
            [
                'S1_000000_IW1_20200101T000000_VV_0000-BURST',
                'S1_000001_IW2_20200101T000001_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20200201T000000_VV_0000-BURST',
                'S1_000002_IW2_20200201T000001_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Number \+ swath \+ polarization identifier does not match for .*'):
        # Different swath
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
            ['S1_000000_IW2_20200201T000000_VV_0000-BURST'],
        )

    with pytest.raises(ValueError, match=r'^Number \+ swath \+ polarization identifier does not match for .*'):
        # Different pol
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
            ['S1_000000_IW1_20200201T000000_VH_0000-BURST'],
        )

    with pytest.raises(ValueError, match=r'^Scenes must have the same polarization. Polarizations present: VH, VV$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20200101T000000_VV_0000-BURST',
                'S1_000000_IW1_20200101T000000_VH_0000-BURST',
            ],
            [
                'S1_000000_IW1_20200201T000000_VV_0000-BURST',
                'S1_000000_IW1_20200201T000000_VH_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^VH polarization is not currently supported, only VV and HH$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20200101T000000_VH_0000-BURST',
                'S1_000000_IW1_20200101T000000_VH_0000-BURST',
            ],
            [
                'S1_000000_IW1_20200201T000000_VH_0000-BURST',
                'S1_000000_IW1_20200201T000000_VH_0000-BURST',
            ],
        )


def test_validate_bursts_datetimes():
    burst.validate_bursts(
        ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        ['S1_000000_IW1_20200101T000001_VV_0000-BURST'],
    )
    burst.validate_bursts(
        [
            'S1_000000_IW1_20250101T000000_VV_0000-BURST',
            'S1_000001_IW1_20250101T000100_VV_0000-BURST',
            'S1_000002_IW1_20250101T000200_VV_0000-BURST',
        ],
        [
            'S1_000000_IW1_20250101T000201_VV_0000-BURST',
            'S1_000001_IW1_20250101T000300_VV_0000-BURST',
            'S1_000002_IW1_20250101T000401_VV_0000-BURST',
        ],
    )

    with pytest.raises(ValueError, match=r'^Reference scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000201_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000401_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                # Test with reference datetimes unsorted
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000401_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Secondary scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000402_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Secondary scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                # Test with secondary datetimes unsorted
                'S1_000000_IW1_20250101T000402_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000201_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000001_VV_0000-BURST'],
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000200_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000400_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000401_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        # Test with reference and secondary datetimes unsorted
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000200_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000000_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000300_VV_0000-BURST',
                'S1_000001_IW1_20250101T000400_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
        )


def test_load_burst_position(tmpdir):
    product = namedtuple('product', ['bursts'])
    bursts = namedtuple(
        'bursts',
        [
            'numberOfLines',
            'numberOfSamples',
            'firstValidLine',
            'numValidLines',
            'firstValidSample',
            'numValidSamples',
            'azimuthTimeInterval',
            'rangePixelSize',
            'sensingStop',
        ],
    )

    mock_product = product([bursts(100, 200, 10, 50, 20, 60, 0.1, 0.2, datetime(2020, 1, 1))])
    with patch('hyp3_isce2.burst.load_product') as mock_load_product:
        mock_load_product.return_value = mock_product
        position = burst.load_burst_position('', 0)

    assert position.n_lines == 100
    assert position.n_samples == 200
    assert position.first_valid_line == 10
    assert position.n_valid_lines == 50
    assert position.first_valid_sample == 20
    assert position.n_valid_samples == 60
    assert position.azimuth_time_interval == 0.1
    assert position.range_pixel_size == 0.2
    assert position.sensing_stop == datetime(2020, 1, 1)


def test_evenize():
    length, valid_start, valid_length = burst.evenize(101, 7, 57, 5)
    assert length == 100
    assert valid_start == 10
    assert valid_length == 50

    length, valid_start, valid_length = burst.evenize(100, 5, 55, 5)
    assert length == 100
    assert valid_start == 5
    assert valid_length == 55

    with pytest.raises(ValueError, match=r'.*valid data region.*'):
        burst.evenize(20, 6, 20, 5)


def test_evenly_subset_position():
    input_pos = burst.BurstPosition(101, 101, 11, 20, 11, 20, 1, 1, datetime(2021, 1, 1, 0, 0, 0))
    ml_params = burst.evenly_subset_position(input_pos, 2, 10)

    assert ml_params.n_lines == 100
    assert ml_params.n_samples == 100
    assert ml_params.n_valid_lines == 10
    assert ml_params.n_valid_samples == 18
    assert ml_params.first_valid_line == 20
    assert ml_params.first_valid_sample == 12
    assert ml_params.azimuth_time_interval == input_pos.azimuth_time_interval
    assert ml_params.range_pixel_size == input_pos.range_pixel_size
    assert ml_params.sensing_stop == datetime(2020, 12, 31, 23, 59, 59)


def test_multilook_position():
    input_pos = burst.BurstPosition(100, 100, 20, 60, 20, 30, 1, 1, datetime(2021, 1, 1, 0, 0, 0))
    output_pos = burst.multilook_position(input_pos, 10, 2)

    assert output_pos.n_lines == 50
    assert output_pos.n_samples == 10
    assert output_pos.first_valid_line == 10
    assert output_pos.n_valid_lines == 30
    assert output_pos.first_valid_sample == 2
    assert output_pos.n_valid_samples == 3
    assert output_pos.azimuth_time_interval == input_pos.azimuth_time_interval * 2
    assert output_pos.range_pixel_size == input_pos.range_pixel_size * 10
    assert output_pos.sensing_stop == input_pos.sensing_stop


def test_safely_multilook(tmpdir):
    image_path = str(tmpdir / 'image')
    make_test_image(image_path)
    pos = burst.BurstPosition(100, 100, 20, 60, 20, 60, 0.1, 0.1, datetime(2021, 1, 1, 0, 0, 0))
    burst.safely_multilook(image_path, pos, 5, 5)
    _, multilooked_array = utils.load_isce2_image(f'{image_path}.multilooked')
    assert multilooked_array.shape == (20, 20)

    golden_array = np.zeros(multilooked_array.shape, dtype=np.float32)
    golden_array[4:16, 4:16] = 1
    assert np.all(multilooked_array == golden_array)


def test_multilook_radar_merge_inputs(tmpdir):
    paths = [
        'fine_interferogram/IW1/burst_01.int',
        'geom_reference/IW1/lat_01.rdr',
        'geom_reference/IW1/lon_01.rdr',
        'geom_reference/IW1/los_01.rdr',
    ]
    tmp_paths = [Path(tmpdir) / x for x in paths]
    for path in tmp_paths:
        make_test_image(str(path))

    mock_position = burst.BurstPosition(100, 100, 20, 60, 20, 60, 0.1, 0.1, datetime(2021, 1, 1, 0, 0, 0))
    with patch('hyp3_isce2.burst.load_burst_position') as mock_load_burst_position:
        mock_load_burst_position.return_value = mock_position
        output = burst.multilook_radar_merge_inputs(1, 5, 2, base_dir=tmpdir)

    assert output.n_lines == 50
    assert output.n_samples == 20

    multilooked = [x.parent / f'{x.stem}.multilooked{x.suffix}' for x in tmp_paths]
    for file in multilooked:
        assert file.exists()
