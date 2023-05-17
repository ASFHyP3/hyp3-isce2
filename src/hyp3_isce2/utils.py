from pathlib import Path

import utm
from osgeo import gdal
from pyproj import CRS


def get_utm_proj(dataset_path: Path):
    dataset = gdal.Open(str(dataset_path))
    longitude = dataset.GetGeoTransform()[0]
    latitude = dataset.GetGeoTransform()[3]
    south = latitude <= 0
    zone = utm.from_latlon(latitude, longitude)[2]
    utm_crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': south})
    dataset = None
    return utm_crs
