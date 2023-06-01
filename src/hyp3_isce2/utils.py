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
        return g2, g1
    return g1, g2
