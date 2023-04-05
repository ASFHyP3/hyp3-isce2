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

import numpy as np
import rasterio
from dem_stitcher.stitcher import stitch_dem
from lxml import etree
from shapely.geometry import box

DEM_RESOLUTION = 0.0002777777777777777775


def tag_dem_xml_as_ellipsoidal(dem_path: Path) -> str:
    xml_path = str(dem_path) + '.xml'
    assert Path(xml_path).exists()
    tree = etree.parse(xml_path)
    root = tree.getroot()

    y = etree.Element("property", name='reference')
    etree.SubElement(y, "value").text = "WGS84"
    etree.SubElement(y, "doc").text = "Geodetic datum"

    root.insert(0, y)
    with open(xml_path, 'wb') as file:
        file.write(etree.tostring(root, pretty_print=True))
    return xml_path


def fix_image_xml(isce_raster_path: str) -> str:
    isce_apps_path = site.getsitepackages()[0] + '/isce/applications'
    fix_cmd = [f'{isce_apps_path}/fixImageXml.py', '-i', str(isce_raster_path), '--full']
    fix_cmd_line = ' '.join(fix_cmd)
    subprocess.check_call(fix_cmd_line, shell=True)
    return isce_raster_path


def buffer_extent(extent: list, buffer: float) -> list:
    extent_geo = box(*extent)
    extent_buffered = list(extent_geo.buffer(buffer).bounds)
    return [
        np.floor(extent_buffered[0]),
        np.floor(extent_buffered[1]),
        np.ceil(extent_buffered[2]),
        np.ceil(extent_buffered[3])
    ]


def download_dem_for_isce2(
        extent: list,
        dem_name: str = 'glo_30',
        dem_dir: Path = None,
        buffer: float = .4) -> dict:
    # FIXME: second ymin should be ymax in `extent` description; open PR to original repo as well
    # TODO: update docstr format
    """
    Parameters
    ----------
    extent : list
        [xmin, ymin, xmax, ymin] for epsg:4326 (i.e. (x, y) = (lon, lat))
    dem_name : str, optional
        See names in `dem_stitcher`
    dem_dir : Path, optional
    buffer : float, optional
        In degrees, by default .1, which is about 11 km at equator
    Returns
    -------
    dict
    """
    dem_dir = dem_dir or Path('.')
    dem_dir.mkdir(exist_ok=True, parents=True)

    extent_buffered = buffer_extent(extent, buffer)

    dem_array, dem_profile = stitch_dem(
        extent_buffered,
        dem_name,
        dst_ellipsoidal_height=True,
        dst_area_or_point='Point',
        n_threads_downloading=5,
        # ensures square resolution
        dst_resolution=DEM_RESOLUTION,
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

    dem_xml = tag_dem_xml_as_ellipsoidal(dem_path)
    fix_image_xml(dem_xml)

    # TODO correct return value format?
    return {'extent_buffered': extent_buffered, 'dem_path': dem_path}
