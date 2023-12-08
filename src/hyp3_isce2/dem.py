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
import os
import json
import subprocess
from pathlib import Path

import dem_stitcher
import numpy as np
import rasterio
from lxml import etree
from shapely import geometry
from shapely.geometry import box, mapping, shape


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
    cmd = ['fixImageXml.py', '-i', xml_path, '--full']
    subprocess.run(cmd, check=True)


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
    return np.round(distance_degrees_lon, 15), np.round(distance_degrees_lat, 15)


def box2dict(bbox):
    """

    Args:
        bbox: tuple(minlon,minlat,maxlon,maxlat)

    Returns:
        dict{{'type': 'Polygon', 'coordinates':

    """

    return {'type': 'Polygon',
            'coordinates': [
                [[bbox[0], bbox[3]],
                 [bbox[0], bbox[1]],
                 [bbox[2], bbox[1]],
                 [bbox[2], bbox[3]],
                 [bbox[0], bbox[3]]
                 ]
            ]
            }


def split_geometry_on_antimeridian(geometry: dict):
    geometry_as_bytes = json.dumps(geometry).encode()
    cmd = ['ogr2ogr', '-wrapdateline', '-datelineoffset', '20', '-f', 'GeoJSON', '/vsistdout/', '/vsistdin/']
    geojson_str = subprocess.run(cmd, input=geometry_as_bytes, stdout=subprocess.PIPE, check=True).stdout
    return json.loads(geojson_str)['features'][0]['geometry']


def get_dem_for_extent(
        extent: list,
        dem_name: str = 'glo_30',
        dem_path: Path = None,
        buffer: float = .4,
        resample_20m: bool = False):
    extent_buffered = buffer_extent(extent, buffer)
    if resample_20m:
        res_degrees = distance_meters_to_degrees(20.0, extent[1])
        dem_array, dem_profile = dem_stitcher.stitch_dem(
            extent_buffered,
            dem_name,
            dst_ellipsoidal_height=True,
            dst_area_or_point='Point',
            n_threads_downloading=5,
            resample_20m=res_degrees
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

    with rasterio.open(dem_path, 'w', **dem_profile) as ds:
        ds.write(dem_array, 1)

    return dem_path


def download_dem_for_isce2_new(
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
    dem_dir = dem_dir or Path('.')
    dem_dir.mkdir(exist_ok=True, parents=True)

    extent_dict = box2dict(extent)
    split_extent = split_geometry_on_antimeridian(extent_dict)

    split_polys = geometry.shape(split_extent)

    polys = [poly for poly in split_polys.geoms]

    # extent_buffered = buffer_extent(extent, buffer)
    file_lst =[]
    for i in range(len(polys)):
        tmp_path = get_dem_for_extent(list(polys[i].bounds), dem_name=dem_name, dem_path=Path(f'tmp_dem_{i}'), buffer=buffer, resample_20m = resample_20m)
        file_lst.append(tmp_path)

    # mosaic the tem_dem files in the file_lst
   
    if resample_20m:
        dem_path = dem_dir / 'full_res_geocode.dem.wgs84'
    else:
        dem_path = dem_dir / 'full_res.dem.wgs84'

    xml_path = tag_dem_xml_as_ellipsoidal(dem_path)
    fix_image_xml(xml_path)

    return dem_path


def get_correct_extent(extent: list):
    if abs(extent[0] - extent[2]) > 180.0 and extent[0] * extent[2] < 0.0:
        # the extent crosses over antimeridian
        tmp = extent[0] + 360.0
        extent[0] = extent[2]
        extent[2] = tmp
    return extent

def get_correct_polygon(poly):
    poly_mapped = mapping(poly)
    poly_coordinates = poly_mapped['coordinates'][0]
    lst = list(poly_coordinates)
    xe = [x[0] for x in lst]
    xe = np.array(xe)

    ye = [y[1] for y in lst]
    ye = np.array(ye)

    xe_min = xe.min()
    xe_max = xe.max()

    correct_poly = poly

    if abs( xe_min - xe_max) > 180.0 and xe_min * xe_max < 0.0:
        # the extent crosses over antimeridian
        lst=[]
        for i in range(len(xe)):
            if xe[i] < 0:
                xe[i] = xe[i] + 360.
            lst.append((xe[i],ye[i]))

        poly_dict = {'type': 'Polygon', 'coordinates': ((tuple(lst)),)}
        correct_poly = shape(poly_dict)

    return correct_poly


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
    dem_dir = dem_dir or Path('.')
    dem_dir.mkdir(exist_ok=True, parents=True)

    correct_extent = get_correct_extent(extent)

    extent_buffered = buffer_extent(correct_extent, buffer)

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


def shift_antimeridian_lon_rdr(lon_rdr: str):
    os.system(f'cp {lon_rdr} {lon_rdr}_org')
    os.system(f'cp {lon_rdr}.xml {lon_rdr}_org.xml')
    with rasterio.open(lon_rdr) as src_ds:
        profile = src_ds.profile
        data  = src_ds.read(1)
        xe_min = data.min()
        xe_max = data.max()
        if abs(xe_min - xe_max) > 180 and xe_min * xe_max < 0:
            data[data < 0] = data[data < 0] + 360.

    with rasterio.open(lon_rdr,'w',**profile) as dst_ds:
        dst_ds.write(data,1)

def convert_geotiff_2_isce(tiff_file, isce_file):
    ds = rasterio.open(tiff_file)
    data = ds.read()
    profile =ds.profile

    profile['driver'] = 'ISCE'

    # remove keys that do not work with ISCE gdal format
    for key in ['blockysize', 'interleave', 'tiled']:
        del profile[key]

    with rasterio.open(isce_file, 'w', **profile) as f:
        f.write(data[0], 1)

    return isce_file