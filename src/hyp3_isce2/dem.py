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
import pyproj
import rasterio
from lxml import etree
from shapely.geometry import box

DEM_RESOLUTION = 0.0002777777777777777775


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


def utm_from_lon_lat(lon: float, lat: float) -> int:
    """Calculate the UTM EPSG code for a given location (lon,lat)
    Args:
        lon: longitude of the location
        lat: latitude of the location
    Returns:
        UTM EPSG code
     """
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone


def get_correct_extent(extent):
    """determine if the extent crossover antimeridian, return correct extent for extent that crossover antimeridian
    Args:
        extent: A list [xmin, ymin, xmax, ymax] for epsg:4326 (i.e. (x, y) = (lon, lat))
    Returns:
        correct extent which assures that longitude of the right side of extent
        is larger than the longitude of the left side of the extent
        """
    if extent[0]*extent[2] < 0 and abs(extent[0]-extent[2]):
        if extent[2] < 0:
            extent[2] += 360.
        if extent[0] < 0:
            extent[0] += 360.
    return extent


def get_dem_resolution(extent: list, pixel_size: float = 80.0):
    """Calculate the pixel spacing in WGS84 for a given pixel spacing in UTM
    Args:
        extent: A list [xmin, ymin, xmax, ymax] for epsg:4326 (i.e. (x, y) = (lon, lat))
        pixel_size: pixel spacing of output product
    Returns:
        The pixel spacing in WGS84 for the given pixel spacing in UTM
        """
    # coordinates of the center pixel

    # deal with extent crossing over antimeridian
    extent = get_correct_extent(extent)

    lon_ul = (extent[0] + extent[2])/2.0
    lat_ul = (extent[1] + extent[3])/2.0

    epsg_code = utm_from_lon_lat(lon_ul, lat_ul)
    myprj1 = pyproj.Transformer.from_crs(4326, epsg_code, always_xy=True)
    myprj2 = pyproj.Transformer.from_crs(epsg_code, 4326, always_xy=True)

    x_ul, y_ul = myprj1.transform(lon_ul, lat_ul)
    lon_ur, lat_ur = myprj2.transform(x_ul + pixel_size, y_ul)
    lon_ll, lat_ll = myprj2.transform(x_ul, y_ul - pixel_size)
    lon_lr, lat_lr = myprj2.transform(x_ul + pixel_size, y_ul - pixel_size)

    # envelope of the pixel in WGS84
    envelope = [min(lon_ul, lon_ll), min(lat_ll, lat_lr), max(lon_ur, lon_lr), max(lat_ul, lat_ur)]

    # adjust the lon values for the pixel crossing over antimeridian
    envelope = get_correct_extent(envelope)

    return max(abs(envelope[2]-envelope[0]), abs(envelope[3] - envelope[1]))


def download_dem_for_isce2(
        extent: list,
        dem_name: str = 'glo_30',
        pixel_size: float = 80.0,
        dem_dir: Path = None,
        buffer: float = .4) -> Path:
    """Download the given DEM for the given extent.

    Args:
        extent: A list [xmin, ymin, xmax, ymax] for epsg:4326 (i.e. (x, y) = (lon, lat)).
        dem_name: One of the names from `dem_stitcher`.
        pixel_size: pixel spacing of output product
        dem_dir: The output directory.
        buffer: The extent buffer in degrees, by default .4, which is about 44 km at the equator
                (or about 2.5 bursts at the equator).
    Returns:
        The path to the downloaded DEM.
    """
    dem_dir = dem_dir or Path('.')
    dem_dir.mkdir(exist_ok=True, parents=True)

    extent_buffered = buffer_extent(extent, buffer)
    dem_array, dem_profile = dem_stitcher.stitch_dem(
        extent_buffered,
        dem_name,
        dst_ellipsoidal_height=True,
        dst_area_or_point='Point',
        n_threads_downloading=5,
        dst_resolution=get_dem_resolution(extent, pixel_size)
    )

    dem_array[np.isnan(dem_array)] = 0.

    dem_profile['nodata'] = None
    dem_profile['driver'] = 'ISCE'

    # remove keys that do not work with ISCE gdal format
    for key in ['blockxsize', 'blockysize', 'compress', 'interleave', 'tiled']:
        del dem_profile[key]

    dem_path = dem_dir / 'full_res.dem.wgs84'
    with rasterio.open(dem_path, 'w', **dem_profile) as ds:
        ds.write(dem_array, 1)

    xml_path = tag_dem_xml_as_ellipsoidal(dem_path)
    fix_image_xml(xml_path)

    return dem_path
