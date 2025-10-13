"""Create and apply a water body mask"""

import subprocess
from pathlib import Path

import numpy as np
from osgeo import gdal


gdal.UseExceptions()

TILE_PATH = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/WATER_MASK/TILES/'


def get_projection_window(filename: Path | str):
    """Get the projection window info from a GeoTIFF, i.e. [ulx, uly, lrx, lry, xRes, yRes].

    Args:
        filename: The path to the input image.
    """
    ds = gdal.Open(str(filename))
    geotransform = ds.GetGeoTransform()
    pixel_size_x = geotransform[1]
    pixel_size_y = geotransform[5]
    x_min = geotransform[0]
    y_max = geotransform[3]
    x_max = x_min + pixel_size_x * ds.RasterXSize
    y_min = y_max + pixel_size_y * ds.RasterYSize
    return x_min, y_max, x_max, y_min, pixel_size_x, pixel_size_y


def get_corners(filename):
    """Get all four corners of the given image: [upper_left, bottom_left, upper_right, bottom_right].

    Args:
        filename: The path to the input image.
    """
    x_min, y_max, x_max, y_min, _, _ = get_projection_window(filename)
    upper_left = [x_min, y_max]
    bottom_left = [x_min, y_min]
    upper_right = [x_max, y_max]
    bottom_right = [x_max, y_min]
    return [upper_left, bottom_left, upper_right, bottom_right]


def coord_to_tile(coord: tuple[float, float]) -> str:
    """Get the filename of the tile which encloses the inputted coordinate.

    Args:
        coord: The (lon, lat) tuple containing the desired coordinate.
    """
    lat_rounded = np.floor(coord[1] / 5) * 5
    lon_rounded = np.floor(coord[0] / 5) * 5
    if lat_rounded >= 0:
        lat_part = 'n' + str(int(lat_rounded)).zfill(2)
    else:
        lat_part = 's' + str(int(np.abs(lat_rounded))).zfill(2)
    if lon_rounded >= 0:
        lon_part = 'e' + str(int(lon_rounded)).zfill(3)
    else:
        lon_part = 'w' + str(int(np.abs(lon_rounded))).zfill(3)
    return lat_part + lon_part + '.tif'


def get_tiles(filename: str) -> list[str]:
    """Get the AWS vsicurl path's to the tiles necessary to cover the inputted file.

    Args:
        filename: The path to the input file (needs to be in EPSG:4326).
    """
    tiles = []
    corners = get_corners(filename)

    # Handle high latitude cases where SLCs may be wider than the tiles
    width = corners[2][0] - corners[0][0]
    if width > 5:
        corners.extend(
            [
                [corners[0][0] + width / 2, corners[0][1]],
                [corners[0][0] + width / 2, corners[1][1]],
            ]
        )

    for corner in corners:
        tile = TILE_PATH + coord_to_tile(corner)
        if tile not in tiles:
            tiles.append(tile)
    return tiles


def create_water_mask(
    input_image: str,
    output_image: str,
    gdal_format='ISCE',
    tmp_path: Path = Path(),
):
    """Create a water mask GeoTIFF with the same geometry as a given input GeoTIFF

    The water masks are assembled from OpenStreetMap and ESA WorldCover data.

    Shoreline data is unbuffered and pixel values of 1 indicate land touches the pixel and 0 indicates there is no
    land in the pixel.

    Args:
        input_image: Path for the input GDAL-compatible image (needs to be in EPSG:4326).
        output_image: Path for the output image
        gdal_format: GDAL format name to create output image as
        tmp_path: An optional path to a temporary directory for temp files.
    """
    tiles = get_tiles(input_image)

    if len(tiles) < 1:
        raise ValueError(f'No water mask tiles found for {tiles}.')

    translate_output_path = str(tmp_path / 'merged.tif')
    translate_input_path = tiles[0]

    x_min, y_max, x_max, y_min, x_res, y_res = get_projection_window(input_image)
    projwin = [str(c) for c in [x_min, y_max, x_max, y_min]]

    # This is WAY faster than using gdal_merge, because of course it is.
    if len(tiles) > 1:
        translate_input_path = str(tmp_path / 'merged.vrt')
        build_vrt_command = ['gdalbuildvrt', translate_input_path] + tiles
        subprocess.run(build_vrt_command, check=True)

    projwin_option = ['-projwin'] + projwin
    pixel_size_option = ['-tr', str(x_res), str(y_res)]
    translate_command = (
        ['gdal_translate'] + pixel_size_option + projwin_option + [translate_input_path, translate_output_path]
    )
    subprocess.run(translate_command, check=True)

    flip_values_command = [
        'gdal_calc.py',
        '-A',
        translate_output_path,
        f'--outfile={output_image}',
        '--calc="numpy.abs((A.astype(numpy.int16) + 1) - 2)"',  # Change 1's to 0's and 0's to 1's.
        f'--format={gdal_format}',
        '--overwrite',
    ]
    subprocess.run(flip_values_command, check=True)
