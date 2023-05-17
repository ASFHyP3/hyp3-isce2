from pathlib import Path

import utm
from osgeo import gdal
from pyproj import CRS


def get_utm_proj(dataset_path: Path) -> CRS:
    """Get the UTM projection of a GDAL dataset that is in EPSG:4326

    Args:
        dataset_path: Path to a GDAL dataset

    Returns:
        pyproj.CRS: UTM projection of the dataset
    """

    dataset = gdal.Open(str(dataset_path))

    crs = CRS(dataset.GetProjection())
    if crs.to_epsg() != 4326:
        raise ValueError(f'Dataset projection (EPSG:{crs.to_epsg()}) is not EPSG:4326')

    longitude = dataset.GetGeoTransform()[0]
    latitude = dataset.GetGeoTransform()[3]
    south = latitude <= 0
    zone = utm.from_latlon(latitude, longitude)[2]
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': south})
    dataset = None
    return utm_crs


def get_extent_and_resolution(dataset_path: Path) -> tuple:
    """Get the extent and resolution of a GDAL dataset
    Args:
        dataset_path: Path to a GDAL dataset
    Returns:
        tuple: (extent, resolution)
    """
    ds = gdal.Open(str(dataset_path))
    x, y = ds.RasterXSize, ds.RasterYSize

    geotransform = ds.GetGeoTransform()
    extent = (
        geotransform[0],
        geotransform[3],
        geotransform[0] + geotransform[1] * x,
        geotransform[3] + geotransform[5] * y,
    )
    ds = None
    x_res = geotransform[1]
    y_res = geotransform[5]
    return extent, (x_res, y_res)
