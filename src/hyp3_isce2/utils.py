import os
import shutil

import isce  # noqa
import isceobj
import numpy as np
from isceobj.Util.ImageUtil.ImageLib import loadImage
from osgeo import gdal

gdal.UseExceptions()


class GDALConfigManager:
    """Context manager for setting GDAL config options temporarily"""

    def __init__(self, **options):
        """
        Args:
            **options: GDAL Config `option=value` keyword arguments.
        """
        self.options = options.copy()
        self._previous_options = {}

    def __enter__(self):
        for key in self.options:
            self._previous_options[key] = gdal.GetConfigOption(key)

        for key, value in self.options.items():
            gdal.SetConfigOption(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self._previous_options.items():
            gdal.SetConfigOption(key, value)


def utm_from_lon_lat(lon: float, lat: float) -> int:
    """Get the UTM zone EPSG code from a longitude and latitude.
    See https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
    for more details on UTM coordinate systems.

    Args:
        lon: Longitude
        lat: Latitude

    Returns:
        UTM zone EPSG code
    """
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone


def extent_from_geotransform(geotransform: tuple, x_size: int, y_size: int) -> tuple:
    """Get the extent and resolution of a GDAL dataset.

    Args:
        geotransform: GDAL geotransform.
        x_size: Number of pixels in the x direction.
        y_size: Number of pixels in the y direction.

    Returns:
        tuple: Extent of the dataset.
    """
    extent = (
        geotransform[0],
        geotransform[3],
        geotransform[0] + geotransform[1] * x_size,
        geotransform[3] + geotransform[5] * y_size,
    )
    return extent


def make_browse_image(input_tif: str, output_png: str) -> None:
    with GDALConfigManager(GDAL_PAM_ENABLED='NO'):
        stats = gdal.Info(input_tif, format='json', stats=True)['stac']['raster:bands'][0]['stats']
        gdal.Translate(
            destName=output_png,
            srcDS=input_tif,
            format='png',
            outputType=gdal.GDT_Byte,
            width=2048,
            strict=True,
            scaleParams=[[stats['minimum'], stats['maximum']]],
        )


def oldest_granule_first(g1, g2):
    if g1[14:29] <= g2[14:29]:
        return g1, g2
    return g2, g1


def resample_to_radar(image_to_resample: str, latin: str, lonin: str, output: str):
    """Resample a geographic image to radar coordinates using a nearest neighbor method.
    The latin and lonin images are used to map from geographic to radar coordinates.

    Args:
        image_to_resample: The path to the image to resample
        latin: The path to the latitude image
        lonin: The path to the longitude image
        output: The path to the output image
    """
    maskim = isceobj.createImage()
    maskim.load(image_to_resample + '.xml')
    latim = isceobj.createImage()
    latim.load(latin + '.xml')
    lonim = isceobj.createImage()
    lonim.load(lonin + '.xml')
    mask = np.fromfile(image_to_resample, maskim.toNumpyDataType())
    lat = np.fromfile(latin, latim.toNumpyDataType())
    lon = np.fromfile(lonin, lonim.toNumpyDataType())
    mask = np.reshape(mask, [maskim.coord2.coordSize, maskim.coord1.coordSize])
    startLat = maskim.coord2.coordStart
    deltaLat = maskim.coord2.coordDelta
    startLon = maskim.coord1.coordStart
    deltaLon = maskim.coord1.coordDelta
    lati = np.clip(((lat - startLat) / deltaLat).astype(int), 0, mask.shape[0] - 1)
    loni = np.clip(((lon - startLon) / deltaLon).astype(int), 0, mask.shape[1] - 1)
    cropped = (mask[lati, loni]).astype(maskim.toNumpyDataType())
    cropped = np.reshape(cropped, (latim.coord2.coordSize, latim.coord1.coordSize))
    cropped.tofile(output)
    croppedim = isceobj.createImage()
    croppedim.initImage(output, 'read', cropped.shape[1], maskim.dataType)
    croppedim.renderHdr()


def isce2_copy(in_path: str, out_path: str):
    """Copy an ISCE2 image file and its metadata.

    Args:
        in_path: The path to the input image file (not the xml).
        out_path: The path to the output image file (not the xml).
    """
    image, _, _ = loadImage(in_path)
    clone = image.clone('write')
    clone.setFilename(out_path)
    clone.renderHdr()
    shutil.copy(in_path, out_path)


def image_math(image_a_path: str, image_b_path: str, out_path: str, expression: str):
    """Run ISCE2's ImageMath.py on two images.

    Args:
        image_a_path: The path to the first image (not the xml).
        image_b_path: The path to the second image (not the xml).
        out_path: The path to the output image.
        expression: The expression to pass to ImageMath.py.
    """
    cmd = f"ImageMath.py -e '{expression}' --a={image_a_path} --b={image_b_path} -o {out_path}"
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))
