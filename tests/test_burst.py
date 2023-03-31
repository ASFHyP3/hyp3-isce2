import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import requests
from hyp3_isce2 import burst
from shapely import geometry

URL_BASE = 'https://datapool.asf.alaska.edu/SLC'


@pytest.fixture()
def ref_metadata():
    metadata_path = Path(__file__).parent.absolute() / 'data' / 'reference.xml'
    xml = ET.parse(metadata_path).getroot()
    return xml


@pytest.fixture()
def sec_metadata():
    metadata_path = Path(__file__).parent.absolute() / 'data' / 'secondary.xml'
    xml = ET.parse(metadata_path).getroot()
    return xml


@pytest.fixture()
def ref_burst(ref_metadata):
    safe_url = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85'
    params = burst.BurstParams(safe_url, 'IW2', 'VV', 3)
    burst_metadata = burst.BurstMetadata(ref_metadata, params)
    return burst_metadata


@pytest.fixture()
def sec_burst(sec_metadata):
    safe_url = 'S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11'
    params = burst.BurstParams(safe_url, 'IW2', 'VV', 3)
    burst_metadata = burst.BurstMetadata(sec_metadata, params)
    return burst_metadata


@pytest.fixture()
def tempdir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


def test_create_gcp_df(ref_burst):
    n_bursts = int(ref_burst.annotation.findall('.//burstList')[0].attrib['count'])
    lines_per_burst = int(ref_burst.annotation.findtext('.//{*}linesPerBurst'))

    gcp_df = ref_burst.create_gcp_df()
    assert np.all(gcp_df.columns == ['line', 'pixel', 'latitude', 'longitude', 'height'])
    assert gcp_df.line.min() == 0
    assert gcp_df.line.max() == (n_bursts * lines_per_burst) - 1


def test_create_geometry(ref_burst):
    burst_number = 3
    real_box = (100.509898817751, 37.69349213923167, 101.5989880944895, 38.00276647361588)

    gcp_df = ref_burst.create_gcp_df()
    box = ref_burst.create_geometry(gcp_df)[1]
    assert ref_burst.burst_number == burst_number
    assert np.all([np.isclose(a, b) for a, b in zip(box, real_box)])


@pytest.mark.parametrize(
    'pattern',
    (
        '*SAFE',
        '*SAFE/annotation/*xml',
        '*SAFE/annotation/calibration/calibration*xml',
        '*SAFE/annotation/calibration/noise*xml',
    ),
)
def test_spoof_safe(ref_burst, tempdir, mocker, pattern):
    tempdir_path = Path(tempdir)
    mocker.patch('hyp3_isce2.burst.download_burst', return_value='')
    burst.spoof_safe(requests.Session(), ref_burst, tempdir_path)
    assert len(list(tempdir_path.glob(pattern))) == 1


def test_get_region_of_interest(ref_burst, sec_burst, ref_metadata):
    granule_name = 'S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11'
    burst_pre = burst.BurstMetadata(
        ref_metadata,
        burst.BurstParams(granule_name, 'IW2', 'VV', 2),
    )
    burst_on = burst.BurstMetadata(
        ref_metadata,
        burst.BurstParams(granule_name, 'IW2', 'VV', 3),
    )
    burst_post = burst.BurstMetadata(
        ref_metadata,
        burst.BurstParams(granule_name, 'IW2', 'VV', 4),
    )
    asc = ref_burst.orbit_direction == 'ascending'
    roi = geometry.box(*burst.get_region_of_interest(ref_burst.footprint, sec_burst.footprint, asc))

    assert not roi.intersects(geometry.box(*burst_pre.footprint.bounds))
    assert roi.intersects(geometry.box(*burst_on.footprint.bounds))
    assert not roi.intersects(geometry.box(*burst_post.footprint.bounds))
