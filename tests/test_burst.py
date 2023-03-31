import xml.etree.ElementTree as ET
from pathlib import Path

from hyp3_isce2 import burst

import numpy as np

import pytest

import requests

from shapely import geometry

URL_BASE = 'https://datapool.asf.alaska.edu/SLC'


@pytest.fixture()
def ref_metadata():
    metadata_path = Path(__file__).parent.absolute() / 'test_data' / 'ref_metadata.xml'
    xml = ET.parse(metadata_path).getroot()
    return xml


@pytest.fixture()
def ref_manifest():
    metadata_path = Path(__file__).parent.absolute() / 'test_data' / 'ref_manifest.xml'
    xml = ET.parse(metadata_path).getroot()
    return xml


@pytest.fixture()
def ref_burst(ref_metadata, ref_manifest):
    safe_url = f'{URL_BASE}/SA/S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.zip'
    params = burst.BurstParams(safe_url, 5, 8)
    burst_metadata = burst.BurstMetadata(ref_metadata, ref_manifest, params)
    return burst_metadata


@pytest.fixture()
def sec_metadata():
    metadata_path = Path(__file__).parent.absolute() / 'test_data' / 'sec_metadata.xml'
    xml = ET.parse(metadata_path).getroot()
    return xml


@pytest.fixture()
def sec_manifest():
    metadata_path = Path(__file__).parent.absolute() / 'test_data' / 'sec_manifest.xml'
    xml = ET.parse(metadata_path).getroot()
    return xml


@pytest.fixture()
def sec_burst(sec_metadata, sec_manifest):
    safe_url = f'{URL_BASE}/SA/S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11.zip'
    params = burst.BurstParams(safe_url, 5, 8)
    burst_metadata = burst.BurstMetadata(sec_metadata, sec_manifest, params)
    return burst_metadata


def test_create_gcp_df(ref_burst):
    n_bursts = int(ref_burst.annotation.findall('.//burstList')[0].attrib['count'])
    lines_per_burst = int(ref_burst.annotation.findtext('.//{*}linesPerBurst'))

    gcp_df = ref_burst.create_gcp_df()
    assert np.all(gcp_df.columns == ['line', 'pixel', 'latitude', 'longitude', 'height'])
    assert gcp_df.line.min() == 0
    assert gcp_df.line.max() == (n_bursts * lines_per_burst) - 1


def test_create_geometry(ref_burst):
    burst_number = 8
    real_box = (53.17067752190982, 27.51599975559423, 54.13361604403157, 27.83356711546872)

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
def test_spoof_safe(ref_burst, tmpdir, mocker, pattern):
    tmpdir = Path(tmpdir)
    mocker.patch('isce2_topsapp.burst.download_geotiff', return_value='')
    burst.spoof_safe(requests.Session(), ref_burst, tmpdir)
    assert len(list(tmpdir.glob(pattern))) == 1


def test_get_region_of_interest(ref_burst, sec_burst, ref_metadata, ref_manifest):
    safe_url = f'{URL_BASE}/SA/S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11.zip'
    burst_7 = burst.BurstMetadata(
        ref_metadata,
        ref_manifest,
        burst.BurstParams(safe_url, 5, 7),
    )
    burst_8 = burst.BurstMetadata(
        ref_metadata,
        ref_manifest,
        burst.BurstParams(safe_url, 5, 8),
    )
    burst_9 = burst.BurstMetadata(
        ref_metadata,
        ref_manifest,
        burst.BurstParams(safe_url, 5, 9),
    )
    asc = ref_burst.orbit_direction == 'ascending'
    roi = geometry.box(*burst.get_region_of_interest(ref_burst.footprint, sec_burst.footprint, asc))

    assert roi.intersects(geometry.box(*burst_8.footprint.bounds))
    assert not roi.intersects(geometry.box(*burst_7.footprint.bounds))
    assert not roi.intersects(geometry.box(*burst_9.footprint.bounds))
