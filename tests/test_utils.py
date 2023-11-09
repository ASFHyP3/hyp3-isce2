import os

import numpy as np
from osgeo import gdal

from hyp3_isce2.utils import (
    GDALConfigManager,
    extent_from_geotransform,
    make_browse_image,
    oldest_granule_first,
    resample_to_radar,
    utm_from_lon_lat,
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


def check_correctness_of_resample(mask, lat, lon, geotransform, data_type, outshape):
    x, x_res, y, y_res = geotransform[0], geotransform[1], geotransform[3], geotransform[5]
    rows = len(lat[:, 0])
    cols = len(lat[0, :])
    mask_rows = len(mask[:, 0])
    mask_cols = len(mask[0, :])
    # get corner coordinates
    ul = (x, y)
    lr = (x + x_res * (mask_cols - 1), y + y_res * (mask_rows - 1))
    mask_x_res = (lr[0] - ul[0]) / (cols - 1)
    mask_y_res = (lr[1] - ul[1]) / (rows - 1)

    for row in range(rows):
        for col in range(cols):
            lat[row, col] = y + row * mask_y_res
            lon[row, col] = x + col * mask_x_res

    resampled_image = resample_to_radar(mask, lat, lon, geotransform, data_type, outshape)

    lon_lat_complex = lon + 1j * lat

    for row in range(len(mask[:, 0])):
        for col in range(len(mask[0, :])):
            mask_lat = y + row * y_res
            mask_lon = x + col * x_res
            complex_pos = mask_lon + 1j * mask_lat
            flat_index = (np.abs(lon_lat_complex - complex_pos)).argmin()
            index = flat_index // cols, flat_index % cols
            # Ensure that the 1's in the original mask are mapped to the resampled image.
            if mask[row, col] == 1:
                assert resampled_image[index[0], index[1]] == 1

    return mask, resampled_image


def resample_with_different_case(resample_rows, resample_cols, mask_rows, mask_cols, geotransform):
    lat = np.zeros((resample_rows, resample_cols))
    lon = np.zeros((resample_rows, resample_cols))
    mask = np.zeros((mask_rows, mask_cols))
    np.fill_diagonal(mask, 1)
    mask[0, mask_cols-1] = 1
    mask[mask_rows-1, 0] = 1
    outshape = (resample_rows, resample_cols)
    data_type = np.byte
    return check_correctness_of_resample(mask, lat, lon, geotransform, data_type, outshape)


def test_resample_to_radar():
    geotransform = (10, 1, 0, 15, 0, -1)
    resample_with_different_case(20, 20, 20, 20, geotransform)
    resample_with_different_case(10, 10, 20, 20, geotransform)
    resample_with_different_case(20, 20, 10, 10, geotransform)
    resample_with_different_case(10, 20, 10, 10, geotransform)
    resample_with_different_case(20, 10, 10, 10, geotransform)
    resample_with_different_case(30, 10, 10, 10, geotransform)
