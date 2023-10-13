import os

import numpy as np

from osgeo import gdal

from hyp3_isce2.utils import (
    GDALConfigManager,
    extent_from_geotransform,
    make_browse_image,
    oldest_granule_first,
    utm_from_lon_lat,
    resample_to_radar
)

gdal.UseExceptions()


def test_utm_from_lon_lat():
    assert utm_from_lon_lat(0, 0) == 32631
    assert utm_from_lon_lat(-179, -1) == 32701
    assert utm_from_lon_lat(179, 1) == 32660
    assert utm_from_lon_lat(27, 89) == 32635
    assert utm_from_lon_lat(182, 1) == 32601
    assert utm_from_lon_lat(-182, 1) == 32660
    assert utm_from_lon_lat(-360, -1) == 32731


def test_extent_from_geotransform():
    assert extent_from_geotransform((0, 1, 0, 0, 0, -1), 1, 1) == (0, 0, 1, -1)
    assert extent_from_geotransform((0, 1, 0, 0, 0, -1), 2, 2) == (0, 0, 2, -2)
    assert extent_from_geotransform((0, 1, 0, 0, 0, -1), 1, 3) == (0, 0, 1, -3)


def test_gdal_config_manager():
    gdal.SetConfigOption('OPTION1', 'VALUE1')
    gdal.SetConfigOption('OPTION2', 'VALUE2')

    assert gdal.GetConfigOption('OPTION1') == 'VALUE1'
    assert gdal.GetConfigOption('OPTION2') == 'VALUE2'
    assert gdal.GetConfigOption('OPTION3') is None
    assert gdal.GetConfigOption('OPTION4') is None

    with GDALConfigManager(OPTION2='CHANGED', OPTION3='VALUE3'):
        assert gdal.GetConfigOption('OPTION1') == 'VALUE1'
        assert gdal.GetConfigOption('OPTION2') == 'CHANGED'
        assert gdal.GetConfigOption('OPTION3') == 'VALUE3'
        assert gdal.GetConfigOption('OPTION4') is None

        gdal.SetConfigOption('OPTION4', 'VALUE4')

    assert gdal.GetConfigOption('OPTION1') == 'VALUE1'
    assert gdal.GetConfigOption('OPTION2') == 'VALUE2'
    assert gdal.GetConfigOption('OPTION3') is None
    assert gdal.GetConfigOption('OPTION4') == 'VALUE4'


def test_oldest_granule_first():
    oldest = "S1_249434_IW1_20230511T170732_VV_07DE-BURST"
    latest = "S1_249434_IW1_20230523T170733_VV_8850-BURST"
    assert oldest_granule_first(oldest, latest) == (oldest, latest)
    assert oldest_granule_first(latest, oldest) == (oldest, latest)


def test_make_browse_image():
    input_tif = "tests/data/test_geotiff.tif"
    output_png = "tests/data/test_browse_image2.png"
    make_browse_image(input_tif, output_png)
    assert open(output_png, "rb").read() == open("tests/data/test_browse_image.png", "rb").read()
    os.remove(output_png)




def test_resample_to_radar():

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def back_to_2d(index, cols):
        return index // cols, index % cols

    dim1 = 20
    dim2 = 15
    mask = np.zeros((dim1, dim2))
    np.fill_diagonal(mask, 1)
    lat = np.zeros((dim1, dim2))
    lon = np.zeros((dim1, dim2))
    geotransform = (-136, 0.000555556, 0, 65.0001, 0, -0.000277778)
    type = np.byte
    outshape = (dim1,dim2)

    for i in range(dim1):
        for j in range(dim2):
            lat[i, j] = 65.0001 + i * (-0.000277778 + (-0.000277778 / 2))
            lon[i, j] = -136 + j * (0.000555556 + (0.000555556 / 2))

    resampled_image = resample_to_radar(mask, lat, lon, geotransform, type, outshape)

    for i in range(dim1):
        for j in range(dim2):
            mask_lat = 65.0001 + i * -0.000277778
            mask_lon = -136 + j * 0.000555556
            flat_lat_index = find_nearest(lat, mask_lat)
            lat_index = back_to_2d(flat_lat_index, dim2)
            flat_lon_index = find_nearest(lon, mask_lon)
            lon_index = back_to_2d(flat_lon_index, dim2)
            if mask[i, j] == 1:
                assert resampled_image[lat_index[0], lon_index[1]] == 1