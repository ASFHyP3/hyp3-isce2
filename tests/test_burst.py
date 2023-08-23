from pathlib import Path
from re import match
from unittest.mock import patch

import asf_search
import pytest
from lxml import etree
from shapely import geometry

from hyp3_isce2 import burst


URL_BASE = 'https://datapool.asf.alaska.edu/SLC'
REF_DESC = burst.BurstParams('S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85', 'IW2', 'VV', 3)
SEC_DESC = burst.BurstParams('S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11', 'IW2', 'VV', 3)
REF_ASC = burst.BurstParams('S1A_IW_SLC__1SDV_20200608T142544_20200608T142610_032927_03D069_14F4', 'IW1', 'VV', 1)
SEC_ASC = burst.BurstParams('S1A_IW_SLC__1SDV_20200620T142544_20200620T142611_033102_03D5B7_8F1B', 'IW1', 'VV', 1)


def load_metadata(metadata):
    metadata_path = Path(__file__).parent.absolute() / 'data' / metadata
    xml = etree.parse(metadata_path).getroot()
    return xml


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
    """
    Test that the region of interest is correctly calculated for a given burst pair.
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


def test_get_product_name():

    reference_name = "S1_136231_IW2_20200604T022312_VV_7C85-BURST"
    secondary_name = "S1_136231_IW2_20200616T022313_VV_5D11-BURST"

    name_20m = burst.get_product_name(reference_name, secondary_name, pixel_spacing=20.0)
    name_80m = burst.get_product_name(reference_name, secondary_name, pixel_spacing=80)

    assert match("[A-F0-9]{4}", name_20m[-4:]) is not None
    assert match("[A-F0-9]{4}", name_80m[-4:]) is not None

    assert name_20m.startswith('S1_136231_IW2_20200604_20200616_VV_INT20')
    assert name_80m.startswith('S1_136231_IW2_20200604_20200616_VV_INT80')


def mock_asf_search_results(
        slc_name: str,
        subswath: str,
        polarization: str,
        burst_index: int) -> asf_search.ASFSearchResults:
    product = asf_search.ASFProduct()
    product.umm = {'InputGranules': [slc_name]}
    product.properties.update({
        'burst': {'subswath': subswath, 'burstIndex': burst_index},
        'polarization': polarization,
    })
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


def test_validate_bursts():
    burst.validate_bursts(
        'S1_030349_IW1_20230808T171601_VV_4A37-BURST',
        'S1_030349_IW1_20230820T171602_VV_5AC3-BURST'
    )
    with pytest.raises(ValueError, match=r'.*polarizations are not the same.*'):
        burst.validate_bursts(
            'S1_215032_IW2_20230802T144608_VV_7EE2-BURST',
            'S1_215032_IW2_20230721T144607_HH_B3FA-BURST'
        )
    with pytest.raises(ValueError, match=r'.*burst IDs are not the same.*'):
        burst.validate_bursts(
            'S1_030349_IW1_20230808T171601_VV_4A37-BURST',
            'S1_030348_IW1_20230820T171602_VV_5AC3-BURST'
        )
    with pytest.raises(ValueError, match=r'.*only VV and HH.*'):
        burst.validate_bursts(
            'S1_030349_IW1_20230808T171601_VH_4A37-BURST',
            'S1_030349_IW1_20230820T171602_VH_5AC3-BURST'
        )
