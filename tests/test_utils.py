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


def foo(mask, lat, lon, geotransform, type, outshape):

    x, x_res, y, y_res = geotransform[0], geotransform[1], geotransform[3], geotransform[5]

    rows = len(lat[:, 0])
    cols = len(lat[0, :])

    for row in range(len(lat[:, 0])):
        for col in range(len(lat[0, :])):
            lat[row, col] = y + row * y_res
            lon[row, col] = x + col * x_res

    resampled_image = resample_to_radar(mask, lat, lon, geotransform, type, outshape)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def back_to_2d(index, cols):
        return index // cols, index % cols

    lon_lat_complex = lon + 1j * lat

    for row in range(len(mask[:, 0])):
        for col in range(len(mask[0, :])):
            mask_lat = y + row * y_res
            mask_lon = x + col * x_res
            flat_index = find_nearest(lon_lat_complex, mask_lon + 1j * mask_lat)
            index = back_to_2d(flat_index, cols)
            if mask[row, col] == 1:
                assert resampled_image[index[0], index[1]] == 1

    return mask, resampled_image

def test_resample_to_radar():

    rows_1 = 20
    cols_1 = 15
    type = np.byte
    outshape_1 = (rows_1, cols_1)
    lat_1 = np.zeros((rows_1, cols_1))
    lon_1 = np.zeros((rows_1, cols_1))
    mask_1 = np.zeros((rows_1, cols_1))
    np.fill_diagonal(mask_1, 1)
    mask_1[0, 14] = 1
    mask_1[19, 0] = 1

    # rows_2 = 20
    # cols_2 = 10
    # type = np.byte
    # outshape_2 = (rows_2, cols_2)
    # lat_2 = np.zeros((rows_2, cols_2))
    # lon_2 = np.zeros((rows_2, cols_2))
    # mask_2 = np.zeros((20, 20))
    # np.fill_diagonal(mask_2, 1)
    # mask_2[0, 19] = 1
    # mask_2[19, 0] = 1

    geotransform = (x := 10, x_res := 1, 0, y := 15, 0, y_res := -1)

    print("input", mask_1)
    m, r = foo(mask_1, lat_1, lon_1, geotransform, type, outshape_1)
    print("output", r)