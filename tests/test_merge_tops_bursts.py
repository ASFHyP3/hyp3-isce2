"""Tests for hyp3_isce2.merge_tops_bursts module, use single quotes"""
from datetime import datetime
from unittest.mock import patch

import asf_search
import pytest

import hyp3_isce2.burst as burst_utils
import hyp3_isce2.merge_tops_bursts as merge


# TODO combine with test_burst.py's version
def mock_asf_search_results(
    slc_name: str, subswath: str, polarization: str, burst_index: int
) -> asf_search.ASFSearchResults:
    product = asf_search.ASFProduct()
    product.umm = {'InputGranules': [slc_name]}
    product.properties.update(
        {
            'burst': {'subswath': subswath, 'burstIndex': burst_index, 'relativeBurstID': burst_index - 1},
            'polarization': polarization,
            'url': 'https://foo.com/bar/baz.zip',
            'startTime': '2020-01-01T00:00:00.0000Z',
            'pathNumber': 1,
        }
    )
    results = asf_search.ASFSearchResults([product])
    results.searchComplete = True
    return results


@pytest.fixture
def burst_product(test_merge_dir):
    product_path = list(test_merge_dir.glob('*'))[0]
    product = merge.BurstProduct(
        granule='bar',
        reference_date=datetime(2020, 6, 4, 2, 23, 15),
        secondary_date=datetime(2020, 6, 16, 2, 23, 16),
        burst_id=0,
        swath='IW2',
        polarization='VV',
        burst_number=1,
        product_path=product_path,
        n_lines=377,
        n_samples=1272,
        range_looks=20,
        azimuth_looks=4,
        first_valid_line=8,
        n_valid_lines=363,
        first_valid_sample=9,
        n_valid_samples=1220,
        az_time_interval=0.008222225199999992,
        rg_pixel_size=46.59124229430646,
        start_utc=datetime(2020, 1, 1, 0, 0),
        stop_utc=datetime(2020, 6, 4, 2, 23, 18, 795712),
        relative_orbit=1,
    )
    return product


def test_to_burst_params(burst_product):
    assert burst_product.to_burst_params() == burst_utils.BurstParams('bar', 'IW2', 'VV', 1)


def test_get_burst_metadata(test_merge_dir, burst_product):
    product_path = list(test_merge_dir.glob('*'))[0]

    with patch('hyp3_isce2.merge_tops_bursts.asf_search.granule_search') as mock_search:
        mock_search.return_value = mock_asf_search_results('bar', 'IW2', 'VV', 1)
        product = merge.get_burst_metadata([product_path])[0]

    assert product == burst_product
