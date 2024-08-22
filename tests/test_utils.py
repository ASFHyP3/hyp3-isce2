import filecmp
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import isceobj  # noqa
import numpy as np
import pytest
from osgeo import gdal

import hyp3_isce2.utils as utils


gdal.UseExceptions()


def test_check_older_granule_is_reference():
    utils.check_older_granule_is_reference(
        'S1_000000_IW1_20200101T000000_VV_0000-BURST', 'S1_000000_IW1_20200201T000000_VV_0000-BURST'
    )

    utils.check_older_granule_is_reference(
        reference=['S1_000000_IW1_20200101T000000_VV_0000-BURST', 'S1_000001_IW1_20200101T000000_VV_0000-BURST'],
        secondary=['S1_000000_IW1_20200201T000000_VV_0000-BURST', 'S1_000001_IW1_20200201T000000_VV_0000-BURST'],
    )

    with pytest.raises(ValueError, match=r'.* granules must be from one date .*'):
        utils.check_older_granule_is_reference(
            reference=['S1_000000_IW1_20200101T000000_VV_0000-BURST', 'S1_000001_IW1_20200101T000000_VV_0000-BURST'],
            secondary=['S1_000000_IW1_20200201T000000_VV_0000-BURST', 'S1_000001_IW1_20200202T000000_VV_0000-BURST'],
        )

    with pytest.raises(ValueError, match=r'Reference granules must be older .*'):
        utils.check_older_granule_is_reference(
            'S1_000000_IW1_20200201T000000_VV_0000-BURST', 'S1_000000_IW1_20200101T000000_VV_0000-BURST'
        )


def test_utm_from_lon_lat():
    assert utils.utm_from_lon_lat(0, 0) == 32631
    assert utils.utm_from_lon_lat(-179, -1) == 32701
    assert utils.utm_from_lon_lat(179, 1) == 32660
    assert utils.utm_from_lon_lat(27, 89) == 32635
    assert utils.utm_from_lon_lat(182, 1) == 32601
    assert utils.utm_from_lon_lat(-182, 1) == 32660
    assert utils.utm_from_lon_lat(-360, -1) == 32731


def test_extent_from_geotransform():
    assert utils.extent_from_geotransform((0, 1, 0, 0, 0, -1), 1, 1) == (0, 0, 1, -1)
    assert utils.extent_from_geotransform((0, 1, 0, 0, 0, -1), 2, 2) == (0, 0, 2, -2)
    assert utils.extent_from_geotransform((0, 1, 0, 0, 0, -1), 1, 3) == (0, 0, 1, -3)


def test_gdal_config_manager():
    gdal.SetConfigOption('OPTION1', 'VALUE1')
    gdal.SetConfigOption('OPTION2', 'VALUE2')

    assert gdal.GetConfigOption('OPTION1') == 'VALUE1'
    assert gdal.GetConfigOption('OPTION2') == 'VALUE2'
    assert gdal.GetConfigOption('OPTION3') is None
    assert gdal.GetConfigOption('OPTION4') is None

    with utils.GDALConfigManager(OPTION2='CHANGED', OPTION3='VALUE3'):
        assert gdal.GetConfigOption('OPTION1') == 'VALUE1'
        assert gdal.GetConfigOption('OPTION2') == 'CHANGED'
        assert gdal.GetConfigOption('OPTION3') == 'VALUE3'
        assert gdal.GetConfigOption('OPTION4') is None

        gdal.SetConfigOption('OPTION4', 'VALUE4')

    assert gdal.GetConfigOption('OPTION1') == 'VALUE1'
    assert gdal.GetConfigOption('OPTION2') == 'VALUE2'
    assert gdal.GetConfigOption('OPTION3') is None
    assert gdal.GetConfigOption('OPTION4') == 'VALUE4'


def test_make_browse_image():
    input_tif = 'tests/data/test_geotiff.tif'
    output_png = 'tests/data/test_browse_image2.png'
    utils.make_browse_image(input_tif, output_png)
    assert open(output_png, 'rb').read() == open('tests/data/test_browse_image.png', 'rb').read()
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

    resampled_image = utils.resample_to_radar(mask, lat, lon, geotransform, data_type, outshape)

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
    mask[0, mask_cols - 1] = 1
    mask[mask_rows - 1, 0] = 1
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


def test_resample_to_radar_io(tmp_path, test_merge_dir):
    out_paths = []
    array = np.ones((10, 10), dtype=np.float32)
    for image_name in ['input', 'lat', 'lon']:
        out_path = str(tmp_path / image_name)
        image = utils.create_image(str(out_path), 10, access_mode='write')
        image.coord1.coordSize = 10
        image.coord2.coordSize = 10
        utils.write_isce2_image_from_obj(image, array)
        out_paths.append(out_path)

    image_to_resample, latin, lonin = out_paths
    output = str(tmp_path / 'output')

    with patch('hyp3_isce2.utils.resample_to_radar') as mock_resample:
        mock_resample.return_value = array
        utils.resample_to_radar_io(image_to_resample, latin, lonin, output)

    assert Path(output).is_file()

    latim, lat = utils.load_isce2_image(latin)
    outputim, outputarray = utils.load_isce2_image(output)
    assert np.all(outputarray == array)


def test_create_image(tmp_path):
    def _check_create_image(path: str, image_subtype: str = 'default'):
        # create an isce image (includes binary, .vrt, and .xml files)
        array = np.array(range(150), dtype=np.float32)
        array = array.reshape(15, 10)
        length, width = array.shape
        out_path = str(tmp_path / 'isce_image_2d')
        utils.write_isce2_image(out_path, array=array)

        # test ifg in create, finalize, and load modes
        path_c = path + '/img_via_create'
        img_c = utils.create_image(
            path_c, width=width, access_mode='write', image_subtype=image_subtype, action='create'
        )
        assert Path(img_c.getFilename()).is_file()

        path_f = path + '/img_via_finalize'
        shutil.copy(out_path, path_f)
        img_f = utils.create_image(
            path_f, width=width, access_mode='read', image_subtype=image_subtype, action='finalize'
        )
        assert Path(img_f.getFilename()).is_file()
        assert Path(img_f.getFilename() + '.vrt').is_file()
        assert Path(img_f.getFilename() + '.xml').is_file()

        path_l = path + '/img_via_load'
        shutil.copy(out_path, path_l)
        shutil.copy(f'{out_path}.vrt', f'{path_l}.vrt')
        shutil.copy(f'{out_path}.xml', f'{path_l}.xml')

        img_l = utils.create_image(path_l, access_mode='load', image_subtype=image_subtype, action='load')
        assert Path(img_l.getFilename()).is_file()
        assert Path(img_f.getFilename() + '.vrt').is_file()
        assert Path(img_f.getFilename() + '.xml').is_file()

    _check_create_image(str(tmp_path), image_subtype='ifg')
    _check_create_image(str(tmp_path), image_subtype='cor')
    _check_create_image(str(tmp_path), image_subtype='unw')
    _check_create_image(str(tmp_path), image_subtype='conncomp')
    _check_create_image(str(tmp_path), image_subtype='default')


def test_write_isce2_image(tmp_path):
    array = np.array(range(150), dtype=np.float32)
    array = array.reshape(15, 10)
    out_path = str(tmp_path / 'isce_image_2d')
    utils.write_isce2_image(out_path, array=array)
    assert Path(out_path).is_file()


def test_write_isce2_image_from_obj(tmp_path):
    def _check_write_isce2_image_from_obj(out_path, bands, length, width):
        image = isceobj.createImage()
        image.initImage(out_path, 'write', width, 'FLOAT', bands)
        image.setLength(length)
        image.setImageType('bil')
        image.createImage()
        utils.write_isce2_image_from_obj(image, array=array)
        assert Path(image.filename).is_file()
        assert Path(image.filename + '.vrt').is_file()
        assert Path(image.filename + '.xml').is_file()

    # 1D array, it shape(width), band=1, length=1
    array = np.array(range(150), dtype=np.float32)
    bands = 1
    length = 1
    width = array.shape[0]
    out_path = str(tmp_path / 'isce_image_1d')
    _check_write_isce2_image_from_obj(out_path, bands, length, width)

    # 2D array, is shape(length, width), band=1
    array = np.array(range(150), dtype=np.float32)
    array = array.reshape(15, 10)
    bands = 1
    length, width = array.shape
    out_path = str(tmp_path / 'isce_image_2d')
    _check_write_isce2_image_from_obj(out_path, bands, length, width)

    # multi-D array, its shape(band,length, width)
    array = np.array(range(150), dtype=np.float32)
    array = array.reshape(3, 5, 10)
    bands, length, width = array.shape
    out_path = str(tmp_path / 'isce_image_md')
    _check_write_isce2_image_from_obj(out_path, bands, length, width)


def test_load_isce2_image(tmp_path):
    in_path = str(tmp_path / 'isce_image_md')
    arrayin = np.array(range(150), dtype=np.float32)
    arrayin = arrayin.reshape(3, 5, 10)
    utils.write_isce2_image(in_path, array=arrayin)
    image_obj, arrayout = utils.load_isce2_image(in_path)
    assert isinstance(image_obj, isceobj.Image.Image.Image)
    assert np.array_equal(arrayin, arrayout)

    in_path = str(tmp_path / 'isce_image_2d')

    arrayin = np.array(range(150), dtype=np.float32)
    arrayin = arrayin.reshape(15, 10)
    utils.write_isce2_image(in_path, array=arrayin)
    image_obj, arrayout = utils.load_isce2_image(in_path)
    assert isinstance(image_obj, isceobj.Image.Image.Image)
    assert np.array_equal(arrayin, arrayout)

    in_path = str(tmp_path / 'isce_image_1d')
    arrayin = np.array(range(150), dtype=np.float32)
    arrayin = arrayin.reshape(1, 150)
    utils.write_isce2_image(in_path, array=arrayin)
    image_obj, arrayout = utils.load_isce2_image(in_path)
    assert isinstance(image_obj, isceobj.Image.Image.Image)
    assert np.array_equal(arrayin, arrayout)


def test_get_geotransform_from_dataset(tmp_path):
    image_path = tmp_path / 'test'
    image = utils.create_image(str(image_path), 100, access_mode='write')

    image.coord1.coordStart = 1
    image.coord1.coordDelta = 10
    image.coord2.coordStart = 11
    image.coord2.coordDelta = 100
    assert utils.get_geotransform_from_dataset(image) == (1, 10, 0, 11, 0, 100)


def test_isce2_copy(tmp_path):
    in_path = str(tmp_path / 'isce_image_2d')
    arrayin = np.array(range(150), dtype=np.float32)
    arrayin = arrayin.reshape(15, 10)
    utils.write_isce2_image(in_path, array=arrayin)
    out_path = str(tmp_path / 'isce_image_2d_copy')
    utils.isce2_copy(in_path, out_path)
    assert filecmp.cmp(in_path, out_path)


def test_image_math(tmp_path):
    in_path1 = str(tmp_path / 'isce_image_2d_1')
    array1 = np.array(range(150), dtype=np.float32)
    array1 = array1.reshape(15, 10)
    utils.write_isce2_image(in_path1, array=array1)

    in_path2 = str(tmp_path / 'isce_image_2d_2')
    array2 = np.array(range(3, 153), dtype=np.float32)
    array2 = array2.reshape(15, 10)
    utils.write_isce2_image(in_path2, array=array2)

    out_path = str(tmp_path / 'isce_image_2d_out')
    utils.image_math(in_path1, in_path2, out_path, 'a + b')
    image_obj_out, arrayout = utils.load_isce2_image(out_path)
    assert np.array_equal(array1 + array2, arrayout)


def test_read_product_metadata(test_merge_dir):
    metafile = list(test_merge_dir.glob('S1_136232*/*.txt'))[0]
    metas = utils.read_product_metadata(metafile)
    assert metas['ReferenceGranule'] == 'S1_136232_IW2_20200604T022315_VV_7C85-BURST'
    assert metas['SecondaryGranule'] == 'S1_136232_IW2_20200616T022316_VV_5D11-BURST'
    assert np.isclose(float(metas['Baseline']), -66.10716474087386)
    assert int(metas['ReferenceOrbitNumber']) == 32861
    assert int(metas['SecondaryOrbitNumber']) == 33036
