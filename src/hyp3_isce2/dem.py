# Copyright 2021-present Caltech
# Modifications Copyright 2023 Alaska Satellite Facility
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import site
import subprocess
import requests
from pathlib import Path
from typing import Tuple

import dem_stitcher
import numpy as np
import rasterio
from lxml import etree
from shapely.geometry import box


def tag_dem_xml_as_ellipsoidal(dem_path: Path) -> str:
    xml_path = str(dem_path) + '.xml'
    assert Path(xml_path).exists()
    root = etree.parse(xml_path).getroot()

    element = etree.Element("property", name='reference')
    etree.SubElement(element, "value").text = "WGS84"
    etree.SubElement(element, "doc").text = "Geodetic datum"

    root.insert(0, element)
    with open(xml_path, 'wb') as file:
        file.write(etree.tostring(root, pretty_print=True))
    return xml_path


def fix_image_xml(xml_path: str) -> None:
    fix_image_path = Path(site.getsitepackages()[0]) / 'isce' / 'applications' / 'fixImageXml.py'
    fix_cmd = ' '.join([str(fix_image_path), '-i', xml_path, '--full'])
    subprocess.check_call(fix_cmd, shell=True)


def buffer_extent(extent: list, buffer: float) -> list:
    extent_geo = box(*extent)
    extent_buffered = list(extent_geo.buffer(buffer).bounds)
    return [
        np.floor(extent_buffered[0]),
        np.floor(extent_buffered[1]),
        np.ceil(extent_buffered[2]),
        np.ceil(extent_buffered[3])
    ]


def distance_meters_to_degrees(distance_meters, latitude):
    """Convert a distance from meters to degrees in longitude and latitude

    Args:
        distance_meters: Arc length in meters.
        latitude: The line of latitude at which the calculation takes place.
    Returns:
        The length in degrees for longitude and latitude, respectively.
    """
    if np.abs(latitude) == 90:
        # np.cos won't return exactly 0, so we must manually raise this exception.
        raise ZeroDivisionError('A Latitude of 90 degrees results in dividing by zero.')
    EARTHS_CIRCUMFERENCE = 40_030_173.59204114  # 2 * pi * 6,371,000.0 (Earth's average radius in meters)
    latitude_circumference = EARTHS_CIRCUMFERENCE * np.cos(np.radians(latitude))
    distance_degrees_lon = distance_meters / latitude_circumference * 360
    distance_degrees_lat = distance_meters / EARTHS_CIRCUMFERENCE * 360
    return (np.round(distance_degrees_lon, 15), np.round(distance_degrees_lat, 15))


def validate_dem_coverage(extent: Tuple[float, float, float, float]):
    """Check whether the DEM covers the area of interest.

    Args:
        extent: The extent of the area of interest. (xmin, ymin, xmax, ymax)

    Returns:
        None
    """

    xmin, ymin, xmax, ymax = extent
    url = f'https://portal.opentopography.org/ajaxRasterJob?action=checkIntersect&opentopoID=OTSDEM.032021.4326.3&x1={xmin}&y1={ymin}&x2={xmax}&y2={ymax}'
    res = requests.get(url=url)
    if res.json()['intersect'] == False:
        raise ValueError(
            f'The extent {extent} is not covered by the COP30 DSM.'
        )


def download_dem_for_isce2(
        extent: list,
        dem_name: str = 'glo_30',
        dem_dir: Path = None,
        buffer: float = .4,
        resample_20m: bool = False
) -> Path:
    """Download the given DEM for the given extent.

    Args:
        extent: A list [xmin, ymin, xmax, ymax] for epsg:4326 (i.e. (x, y) = (lon, lat)).
        dem_name: One of the names from `dem_stitcher`.
        dem_dir: The output directory.
        buffer: The extent buffer in degrees, by default .4, which is about 44 km at the equator
                (or about 2.5 bursts at the equator).
        resample_20m: Whether or not the DEM should be resampled to 20 meters.
    Returns:
        The path to the downloaded DEM.
    """

    validate_dem_coverage(extent=extent)

    dem_dir = dem_dir or Path('.')
    dem_dir.mkdir(exist_ok=True, parents=True)

    extent_buffered = buffer_extent(extent, buffer)

    if resample_20m:
        res_degrees = distance_meters_to_degrees(20.0, extent_buffered[1])
        dem_array, dem_profile = dem_stitcher.stitch_dem(
            extent_buffered,
            dem_name,
            dst_ellipsoidal_height=True,
            dst_area_or_point='Point',
            n_threads_downloading=5,
            dst_resolution=res_degrees
        )
        dem_path = dem_dir / 'full_res_geocode.dem.wgs84'
    else:
        dem_array, dem_profile = dem_stitcher.stitch_dem(
            extent_buffered,
            dem_name,
            dst_ellipsoidal_height=True,
            dst_area_or_point='Point',
            n_threads_downloading=5,
        )
        dem_path = dem_dir / 'full_res.dem.wgs84'

    dem_array[np.isnan(dem_array)] = 0.

    dem_profile['nodata'] = None
    dem_profile['driver'] = 'ISCE'

    # remove keys that do not work with ISCE gdal format
    for key in ['blockxsize', 'blockysize', 'compress', 'interleave', 'tiled']:
        del dem_profile[key]

    with rasterio.open(dem_path, 'w', **dem_profile) as ds:
        ds.write(dem_array, 1)

    xml_path = tag_dem_xml_as_ellipsoidal(dem_path)
    fix_image_xml(xml_path)

    return dem_path
