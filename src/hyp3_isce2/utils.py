def utm_from_lon_lat(lon: float, lat: float) -> int:
    """Get the UTM zone epsg from a longitude and latitude.

    Args:
        lon: Longitude
        lat: Latitude

    Returns:
        UTM zone epsg
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
