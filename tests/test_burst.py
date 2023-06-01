from pathlib import Path

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
    assert burst.get_product_name('A', 'B') == 'AxB'
