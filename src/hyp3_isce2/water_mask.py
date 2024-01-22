"""Create and apply a water body mask"""
from os import system

import shapely
import numpy as np

from osgeo import gdal
from rasterio import open as ropen
from rasterio.mask import mask

gdal.UseExceptions()

TILE_PATH = '/Users/asplayer/data/planet-pbf/global_10m/merge_tiles/'


def get_corners(filename):

    ds = gdal.Warp('tmp.tif', filename, dstSRS='EPSG:4326')
    geoTransform = ds.GetGeoTransform()

    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * ds.RasterXSize
    miny = maxy + geoTransform[5] * ds.RasterYSize

    upper_left = [minx, maxy]
    bottom_left = [minx, miny]
    upper_right = [maxx, maxy]
    bottom_right = [maxx, miny]
    
    return [upper_left, bottom_left, upper_right, bottom_right]


def corner_to_tile(corner):
    lat_part = ''
    lon_part = ''
    lat = corner[1]
    lon = corner[0]
    lat_rounded = np.floor(lat / 5) * 5
    lon_rounded = np.floor(lon / 5) * 5
    if lat_rounded >= 0:
        lat_part = 'n' + str(int(lat_rounded)).zfill(2)
    else:
        lat_part = 's' + str(int(np.abs(lat_rounded))).zfill(2)
    if lon_rounded >= 0:
        lon_part = 'e' + str(int(lon_rounded)).zfill(3)
    else:
        lon_part = 'w' + str(int(np.abs(lon_rounded))).zfill(3)
    return lat_part + lon_part + '.tif'


def get_tiles(filename):
    tiles = []
    corners = get_corners(filename)
    for corner in corners:
            tile = TILE_PATH + corner_to_tile(corner)
            if tile not in tiles:
                tiles.append(tile)
    return tiles


def create_water_mask(input_image: str, output_image: str, gdal_format = 'ISCE'):
    """Create a water mask GeoTIFF with the same geometry as a given input GeoTIFF

    The water mask is assembled from OpenStreetMaps data.

    Shoreline data is unbuffered and pixel values of 1 indicate land touches the pixel and 0 indicates there is no
    land in the pixel.

    Args:
        input_image: Path for the input GDAL-compatible image
        output_image: Path for the output image
        gdal_format: GDAL format name to create output image as
    """
    
    tiles = get_tiles(input_image)

    if len(tiles) < 1:
        raise Exception("No Water Mask Tiles Found")

    pixel_size = gdal.Warp('tmp_px_size.tif', input_image, dstSRS='EPSG:4326').GetGeoTransform()[1]

    # This is WAY faster than using gdal_merge, because of course it is.
    if len(tiles) > 1:
        build_vrt_command = ' '.join(['gdalbuildvrt', 'merged.vrt'] + tiles)
        system(build_vrt_command)
        translate_command = ' '.join(['gdal_translate', 'merged.vrt', 'merged.tif'])
        system(translate_command)

    shapefile_command = ' '.join(['gdaltindex', 'tmp.shp', input_image])
    system(shapefile_command)

    warp_filename = 'merged.tif' if len(tiles) > 1 else tiles[0]

    gdal.Warp(
        'merged_warped.tif',
        warp_filename,
        cutlineDSName='tmp.shp',
        cropToCutline=True,
        xRes=pixel_size,
        yRes=pixel_size,
        targetAlignedPixels=True,
        dstSRS='EPSG:4326',
        format=gdal_format
    )

    flip_values_command = ''.join([
        'gdal_calc.py',
        '-A',
        'merged_warped.tif',
        f'--outfile={output_image}',
        '--calc="numpy.abs((A.astype(numpy.int16) + 1) - 2)"',
        f'--format={gdal_format}'
    ])
    system(flip_values_command)

    print('Water Mask Created...')

    # flip_values_command = f'gdal_calc.py -A merged_warped.tif --outfile={output_image} --calc="numpy.abs((A.astype(numpy.int16) + 1) - 2)" --format={gdal_format}'
    # insar_tops_burst --looks 5x1 --apply-water-mask True S1_078088_IW3_20230705T055038_VV_F183-BURST S1_078088_IW3_20230717T055039_VV_F821-BURST