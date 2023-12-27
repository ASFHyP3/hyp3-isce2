"""Create and apply a water body mask"""
from os import system

import shapely
import numpy as np

from osgeo import gdal
from rasterio import open as ropen
from rasterio.mask import mask

gdal.UseExceptions()

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
            tile = corner_to_tile(corner)
            if tile not in tiles:
                tiles.append(tile)
    return tiles


def clip_water_mask(filepath, outfilepath):

    src = gdal.Warp('tmp_to_4326.tif', filepath, dstSRS='EPSG:4326')
    band = src.GetRasterBand(1)
    arr = band.ReadAsArray()

    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    geometry = [[ulx,lry], [ulx,uly], [lrx,uly], [lrx,lry]]
    roi = [shapely.Polygon(geometry)]

    ds = ropen('merged_warped.tif')
    output = mask(ds, roi, crop = True)[0][0]

    rows = src.RasterYSize
    cols = src.RasterXSize
    driver = src.GetDriver()
    out_ds = driver.Create(outfilepath, cols, rows, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(src.GetGeoTransform())
    out_ds.SetProjection(src.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(output)

    del out_ds, out_band

    return output, arr


def create_water_mask(input_image: str, output_image: str):
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

    print(tiles)

    pixel_size = gdal.Warp('tmp_px_size.tif', input_image, dstSRS='EPSG:4326').GetGeoTransform()[1]

    print(pixel_size)
    
    merge_command = ' '.join(['gdal_merge.py', '-o', 'merged.tif'] + tiles)
    system(merge_command)

    shapefile_command = ' '.join(['gdaltindex', 'tmp.shp', input_image])
    system(shapefile_command)

    warped = gdal.Warp(output_image, 'merged.tif', cutlineDSName='tmp.shp', cropToCutline=True, xRes=pixel_size, yRes=pixel_size, targetAlignedPixels=True, dstSRS='EPSG:4326')

    print('Done.')