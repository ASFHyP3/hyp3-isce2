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
from pathlib import Path

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
    """Get the arc length in degrees for a give distance in meters along lines of longitude and latitude.

    Args:
        distance_meters: The desired arc length in meters.
        latitude: The desired line of latitude.
    Returns:
        The length in degrees for longitude and the given latitude, respectively.
    """
    if latitude == 90:
        # np.cos won't return exactly 0, so we must manually raise this exception.
        raise ZeroDivisionError('A Latitude of 90 degrees results in dividing by zero.')
    EARTHS_CIRCUMFERENCE = 40030173.59204114  # 2 * pi * 6371000.0 (Earth's Radius)
    lat_radians = np.radians(latitude)
    circumference_at_latitude = EARTHS_CIRCUMFERENCE * np.cos(lat_radians)
    distance_degrees_lon = distance_meters / circumference_at_latitude * 360
    distance_degrees_lat = distance_meters / EARTHS_CIRCUMFERENCE * 360
    return (np.round(distance_degrees_lon, 15), np.round(distance_degrees_lat, 15))


def download_dem_for_isce2(
        extent: list,
        dem_name: str = 'glo_30',
        dem_dir: Path = None,
        buffer: float = .4,
        resample: bool = False
) -> Path:
    """Download the given DEM for the given extent.

    Args:
        extent: A list [xmin, ymin, xmax, ymax] for epsg:4326 (i.e. (x, y) = (lon, lat)).
        dem_name: One of the names from `dem_stitcher`.
        dem_dir: The output directory.
        buffer: The extent buffer in degrees, by default .4, which is about 44 km at the equator
                (or about 2.5 bursts at the equator).
    Returns:
        The path to the downloaded DEM.
    """
    dem_dir = dem_dir or Path('.')
    dem_dir.mkdir(exist_ok=True, parents=True)

    extent_buffered = buffer_extent(extent, buffer)

    if resample:
        res_degrees = distance_meters_to_degrees(20.0, extent_buffered[1])
        dem_array, dem_profile = dem_stitcher.stitch_dem(
            extent_buffered,
            dem_name,
            dst_ellipsoidal_height=True,
            dst_area_or_point='Point',
            n_threads_downloading=5,
            dst_resolution=res_degrees
        )
    else:
        dem_array, dem_profile = dem_stitcher.stitch_dem(
            extent_buffered,
            dem_name,
            dst_ellipsoidal_height=True,
            dst_area_or_point='Point',
            n_threads_downloading=5,
        )

    dem_array[np.isnan(dem_array)] = 0.

    dem_profile['nodata'] = None
    dem_profile['driver'] = 'ISCE'

    # remove keys that do not work with ISCE gdal format
    for key in ['blockxsize', 'blockysize', 'compress', 'interleave', 'tiled']:
        del dem_profile[key]

    if resample:
        dem_path = dem_dir / 'full_res_geocode.dem.wgs84'
    else:
        dem_path = dem_dir / 'full_res.dem.wgs84'
    with rasterio.open(dem_path, 'w', **dem_profile) as ds:
        ds.write(dem_array, 1)

    xml_path = tag_dem_xml_as_ellipsoidal(dem_path)
    fix_image_xml(xml_path)

    return dem_path
