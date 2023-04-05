from pathlib import Path
from unittest.mock import patch

import numpy as np
from affine import Affine
from rasterio import CRS

from hyp3_isce2 import localize_dem

DEM_ARRAY = None  # TODO

DEM_PROFILE = {
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


def test_download_dem_for_isce2():
    # TODO use temp dir for dem_dir
    # TODO not sure if this patch will work, may have to import stitch_dem differently in localize_dem
    with patch.object(localize_dem, 'stitch_dem') as mock_stitch_dem:
        mock_stitch_dem.return_value = (DEM_PROFILE, DEM_ARRAY)
        localize_dem.download_dem_for_isce2(
            extent=[-169, 53, -168, 54],
            dem_name='glo_30',
            dem_dir=Path('isce_dem'),
            buffer=0,
        )
        assert mock_stitch_dem.called_once_with(
            [-169, 53, -168, 54],
            'glo_30',
            dst_ellipsoidal_height=True,
            dst_area_or_point='Point',
            n_threads_downloading=5,
            dst_resolution=localize_dem.DEM_RESOLUTION,
        )
        # TODO more asserts


def test_buffer_extent():
    extent1 = [1, 2, 3, 4]
    assert localize_dem.buffer_extent(extent1, 0) == extent1
    assert localize_dem.buffer_extent(extent1, 0.1) == [0, 1, 4, 5]

    extent2 = [-169.7, 53.3, -167.3, 54.7]
    assert localize_dem.buffer_extent(extent2, 0) == [-170, 53, -167, 55]
    assert localize_dem.buffer_extent(extent2, 0.1) == [-170, 53, -167, 55]
    assert localize_dem.buffer_extent(extent2, 0.3) == [-170, 53, -167, 55]
    assert localize_dem.buffer_extent(extent2, 0.4) == [-171, 52, -166, 56]
