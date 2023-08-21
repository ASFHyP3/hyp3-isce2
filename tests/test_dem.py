from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine
from lxml import etree
from pytest import raises
from rasterio import CRS

from hyp3_isce2 import dem

MOCK_DEM_ARRAY = np.ones((3600, 3600), dtype=float)
MOCK_DEM_ARRAY[:, 1000] = np.nan
MOCK_DEM_ARRAY[:, 2000] = 2

MOCK_DEM_PROFILE = {
    'blockxsize': 1024,
    'blockysize': 1024,
    'compress': 'deflate',
    'count': 1,
    'crs': CRS.from_epsg(4326),
    'driver': 'GTiff',
    'dtype': 'float32',
    'height': 3600,
    'interleave': 'band',
    'nodata': np.nan,
    'tiled': True,
    'transform': Affine(
        0.0002777777777777778, 0.0, -169.00020833333335,
        0.0, -0.0002777777777777778, 54.00013888888889
    ),
    'width': 3600,
}


def test_download_dem_for_isce2(tmp_path):
    dem_dir = tmp_path / 'isce2_dem'
    dem_dir.mkdir()

    with patch('dem_stitcher.stitch_dem') as mock_stitch_dem:
        mock_stitch_dem.return_value = (MOCK_DEM_ARRAY, MOCK_DEM_PROFILE)

        dem_path = dem.download_dem_for_isce2(
            extent=[-168.7, 53.2, -168.2, 53.7],
            dem_name='glo_30',
            dem_dir=dem_dir,
            buffer=0,
        )

        mock_stitch_dem.assert_called_once_with(
            [-169, 53, -168, 54],
            'glo_30',
            dst_ellipsoidal_height=True,
            dst_area_or_point='Point',
            n_threads_downloading=5,
        )

        root = etree.parse(str(dem_path) + '.xml').getroot()
        assert root.find("./property[@name='reference']/value").text == 'WGS84'
        assert root.find("./property[@name='reference']/doc").text == 'Geodetic datum'

        with rasterio.open(dem_path, 'r') as ds:
            dem_array = ds.read(1)

        assert np.all(dem_array[:, 0:1000] == 1)
        assert np.all(dem_array[:, 1000] == 0)
        assert np.all(dem_array[:, 1001:2000] == 1)
        assert np.all(dem_array[:, 2000] == 2)
        assert np.all(dem_array[:, 2001:] == 1)


def test_buffer_extent():
    extent1 = [1, 2, 3, 4]
    assert dem.buffer_extent(extent1, 0) == extent1
    assert dem.buffer_extent(extent1, 0.1) == [0, 1, 4, 5]

    extent2 = [-169.7, 53.3, -167.3, 54.7]
    assert dem.buffer_extent(extent2, 0) == [-170, 53, -167, 55]
    assert dem.buffer_extent(extent2, 0.1) == [-170, 53, -167, 55]
    assert dem.buffer_extent(extent2, 0.3) == [-170, 53, -167, 55]
    assert dem.buffer_extent(extent2, 0.4) == [-171, 52, -166, 56]


def test_distance_meters_to_degrees():
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=0) == (0.000179864321184, 0.000179864321184)
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=45) == (0.000254366562405, 0.000179864321184)
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=89.9) == (0.103054717208573, 0.000179864321184)
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=-45) == (0.000254366562405, 0.000179864321184)
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=-89.9) == (0.103054717208573, 0.000179864321184)
    # This is since cos(90) = 0, leading to a divide by zero issue.
    with raises(ZeroDivisionError):
        dem.distance_meters_to_degrees(20, 90)
    with raises(ZeroDivisionError):
        dem.distance_meters_to_degrees(20, -90)


def test_validate_dem_coverage():
    try:
        dem.validate_dem_coverage([-100, 30, -101, 31])  # Valid extent over Texas
    except Exception as e:
        assert False, f'Unexpected exception for valid DEM extent: {e}'
    with raises(ValueError, match=r'.*not covered by the COP30 DSM.*'):
        dem.validate_dem_coverage([45, 40, 49, 41])  # Invalid extent over Azerbaijan
    with raises(ValueError, match=r'.*not covered by the COP30 DSM.*'):
        dem.validate_dem_coverage([-33, 28, -30, 29])  # Invalid extent over the Atlantic Ocean
