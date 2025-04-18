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

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from hyp3lib.dem import prepare_dem_geotiff
from lxml import etree
from osgeo import gdal, ogr
from shapely.geometry import box


gdal.UseExceptions()


def tag_dem_xml_as_ellipsoidal(dem_path: Path) -> str:
    xml_path = str(dem_path) + '.xml'
    assert Path(xml_path).exists()
    root = etree.parse(xml_path).getroot()

    element = etree.Element('property', name='reference')
    etree.SubElement(element, 'value').text = 'WGS84'
    etree.SubElement(element, 'doc').text = 'Geodetic datum'

    root.insert(0, element)
    with open(xml_path, 'wb') as file:
        file.write(etree.tostring(root, pretty_print=True))
    return xml_path


def fix_image_xml(xml_path: str) -> None:
    cmd = ['fixImageXml.py', '-i', xml_path, '--full']
    subprocess.run(cmd, check=True)


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
    return np.round(distance_degrees_lon, 15), np.round(distance_degrees_lat, 15)


def download_dem_for_isce2(extent: tuple[float, float, float, float], dem_path: Path, pixel_size: float) -> Path:
    """Download the given DEM for the given extent.

    Args:
        extent: A list [xmin, ymin, xmax, ymax] for epsg:4326 (i.e. (x, y) = (lon, lat)).
        dem_path: The path to write the DEM to.
        resolution: The resolution of the DEM in meters.

    Returns:
        The path to the downloaded DEM.
    """
    with tempfile.NamedTemporaryFile(suffix='.tif') as tmp_dem:
        prepare_dem_geotiff(
            Path(tmp_dem.name),
            ogr.CreateGeometryFromWkb(box(*extent).wkb),
            epsg_code=4326,
            pixel_size=distance_meters_to_degrees(pixel_size, extent[1])[0],
            buffer_size_in_degrees=0.2,
            height_above_ellipsoid=True,
        )
        gdal.Translate(str(dem_path), tmp_dem.name, format='ISCE')

    xml_path = tag_dem_xml_as_ellipsoidal(dem_path)
    fix_image_xml(xml_path)
    return dem_path
