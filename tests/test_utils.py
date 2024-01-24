import os
from pathlib import Path
import shutil
import numpy as np
import pytest
from osgeo import gdal
import filecmp

from hyp3_isce2.utils import (
    ESA_HOST,
    GDALConfigManager,
    extent_from_geotransform,
    get_esa_credentials,
    make_browse_image,
    oldest_granule_first,
    resample_to_radar,
    resample_to_radar_io,
    utm_from_lon_lat,
    create_image,
    write_isce2_image,
    write_isce2_image_from_obj,
    load_isce2_image,
    get_geotransform_from_dataset,
    isce2_copy,
    image_math,
    load_product,
    read_product_metadata,
)
import isceobj

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


def test_resample_to_radar_io(tmp_path):
    image_to_resample = 'tests/data/test_case/dem/down2_res.dem.wgs84'
    latin = 'tests/data/test_case/geom_reference/IW2/down2_lat_01.rdr'
    lonin = 'tests/data/test_case/geom_reference/IW2/down2_lon_01.rdr'
    output = str(tmp_path / 'output')

    latim, lat = load_isce2_image(latin)

    resample_to_radar_io(image_to_resample, latin, lonin, output)

    assert Path(output).is_file()

    outputim, outputarray = load_isce2_image(output)
    assert outputarray.shape == lat.shape

def test_get_esa_credentials_env(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.setenv('ESA_USERNAME', 'foo')
        m.setenv('ESA_PASSWORD', 'bar')
        m.setenv('HOME', str(tmp_path))
        (tmp_path / '.netrc').write_text(f'machine {ESA_HOST} login netrc_username password netrc_password')

        username, password = get_esa_credentials()
        assert username == 'foo'
        assert password == 'bar'


def test_get_esa_credentials_netrc(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.delenv('ESA_USERNAME', raising=False)
        m.delenv('ESA_PASSWORD', raising=False)
        m.setenv('HOME', str(tmp_path))
        (tmp_path / '.netrc').write_text(f'machine {ESA_HOST} login foo password bar')

        username, password = get_esa_credentials()
        assert username == 'foo'
        assert password == 'bar'


def test_get_esa_credentials_missing(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        m.delenv('ESA_USERNAME', raising=False)
        m.setenv('ESA_PASSWORD', 'env_password')
        m.setenv('HOME', str(tmp_path))
        (tmp_path / '.netrc').write_text('')
        msg = 'Please provide.*'
        with pytest.raises(ValueError, match=msg):
            get_esa_credentials()

    with monkeypatch.context() as m:
        m.setenv('ESA_USERNAME', 'env_username')
        m.delenv('ESA_PASSWORD', raising=False)
        m.setenv('HOME', str(tmp_path))
        (tmp_path / '.netrc').write_text('')
        msg = 'Please provide.*'
        with pytest.raises(ValueError, match=msg):
            get_esa_credentials()


def test_create_image(tmp_path):
    def _check_create_image(path: str, image_subtype: str = 'default'):
        # test ifg in create, finalize, and load
        path_c = path + '/img_via_create'
        img_c = create_image(path_c, width=5, access_mode='write', image_subtype=image_subtype, action='create')
        assert Path(img_c.getFilename()).is_file()

        path_f = path + '/img_via_finalize'
        img_f = create_image(path_f, width=5, access_mode='write', image_subtype=image_subtype, action='finalize')
        assert Path(img_f.getFilename()).is_file()
        assert Path(img_f.getFilename() + '.vrt').is_file()
        assert Path(img_f.getFilename() + '.xml').is_file()

        path_l = path + '/img_via_load'
        shutil.copy(path_f, path_l)
        shutil.copy(f'{path_f}.vrt', f'{path_l}.vrt')
        shutil.copy(f'{path_f}.xml', f'{path_l}.xml')

        img_l = create_image(path_l, access_mode='write', image_subtype=image_subtype, action='load')
        assert Path(img_l.getFilename()).is_file()

    _check_create_image(str(tmp_path), image_subtype='ifg')
    _check_create_image(str(tmp_path), image_subtype='cor')
    _check_create_image(str(tmp_path), image_subtype='unw')
    _check_create_image(str(tmp_path), image_subtype='conncomp')
    _check_create_image(str(tmp_path), image_subtype='default')


def test_write_isce2_image(tmp_path):
    array = np.array(range(150), dtype=np.float32)
    array = array.reshape(15, 10)
    bands = 1
    length, width = array.shape
    out_path = str(tmp_path / 'isce_image_2d')
    write_isce2_image(out_path, array=array, bands=bands,length=length, width=width, mode='write', data_type='FLOAT')
    assert Path(out_path).is_file()


def test_write_isce2_image_from_obj(tmp_path):
    def _check_write_isce2_image_from_obj(out_path, bands, length, width):
        image = isceobj.createImage()
        image.initImage(out_path, 'write', width, 'FLOAT', bands)
        image.setLength(length)
        image.setImageType('bil')
        image.createImage()
        write_isce2_image_from_obj(image, array=array)
        assert Path(image.filename).is_file()
        assert Path(image.filename + '.vrt').is_file()
        assert Path(image.filename + '.xml').is_file()

    # 1D array, it shape(width), band=1, length=1
    array = np.array(range(150), dtype = np.float32)
    bands = 1
    length = 1
    width = array.shape[0]
    out_path = str(tmp_path / 'isce_image_1d')
    _check_write_isce2_image_from_obj(out_path, bands, length, width)

    # 2D array, is shape(length, width), band=1
    array = np.array(range(150), dtype = np.float32)
    array = array.reshape(15, 10)
    bands = 1
    length, width = array.shape
    out_path = str(tmp_path / 'isce_image_2d')
    _check_write_isce2_image_from_obj(out_path, bands, length, width)

    # multi-D array, its shape(band,length, width)
    array = np.array(range(150), dtype = np.float32)
    array = array.reshape(3, 5, 10)
    bands, length, width = array.shape
    out_path = str(tmp_path / 'isce_image_md')
    _check_write_isce2_image_from_obj(out_path, bands, length, width)


def test_load_isce2_image(tmp_path):
    in_path = str(tmp_path / 'isce_image_md')
    arrayin = np.array(range(150), dtype = np.float32)
    arrayin = arrayin.reshape(3, 5, 10)
    bands, length, width = arrayin.shape
    write_isce2_image(in_path, array=arrayin, bands=bands, length=length, width=width, mode='write', data_type='FLOAT')
    image_obj, arrayout = load_isce2_image(in_path)
    assert type(image_obj) == isceobj.Image.Image.Image
    assert np.array_equal(arrayin, arrayout)

    in_path = str(tmp_path / 'isce_image_2d')

    arrayin = np.array(range(150), dtype =  np.float32)
    arrayin = arrayin.reshape(15, 10)
    bands = 1
    length, width = arrayin.shape
    write_isce2_image(in_path, array=arrayin, bands=bands, length=length, width=width, mode='write', data_type='FLOAT')
    image_obj, arrayout = load_isce2_image(in_path)
    assert type(image_obj) == isceobj.Image.Image.Image
    assert np.array_equal(arrayin, arrayout)

    in_path = str(tmp_path / 'isce_image_1d')
    arrayin = np.array(range(150), dtype=np.float32)
    arrayin = arrayin.reshape(1, 150)
    bands = 1
    length, width = arrayin.shape
    write_isce2_image(in_path, array=arrayin, bands=bands, length=length, width=width, mode='write', data_type='FLOAT')
    image_obj, arrayout = load_isce2_image(in_path)
    assert type(image_obj) == isceobj.Image.Image.Image
    assert np.array_equal(arrayin, arrayout)


def test_get_geotransform_from_dataset():
    in_path = 'tests/data/test_case/dem/down2_res.dem.wgs84'
    image_obj, _ = load_isce2_image(in_path)
    assert get_geotransform_from_dataset(image_obj) == (52.9999, 0.0277778, 0, 28.0001, 0, -0.0277778)


def test_isce2_copy(tmp_path):
    in_path = str(tmp_path / 'isce_image_2d')
    arrayin = np.array(range(150), dtype=np.float32)
    arrayin = arrayin.reshape(15, 10)
    bands = 1
    length, width = arrayin.shape
    write_isce2_image(in_path, array=arrayin, bands=bands, length=length, width=width, mode='write', data_type='FLOAT')
    out_path = str(tmp_path / 'isce_image_2d_copy')
    isce2_copy(in_path, out_path)
    assert filecmp.cmp(in_path, out_path)


def test_image_math(tmp_path):
    in_path1 = str(tmp_path / 'isce_image_2d_1')
    array1 = np.array(range(150), dtype=np.float32)
    array1 = array1.reshape(15, 10)
    bands = 1
    length, width = array1.shape
    write_isce2_image(in_path1, array=array1, bands=bands, length=length, width=width, mode='write', data_type='FLOAT')

    in_path2 = str(tmp_path / 'isce_image_2d_2')
    array2 = np.array(range(3, 153), dtype=np.float32)
    array2 = array2.reshape(15, 10)
    bands = 1
    length, width = array2.shape
    write_isce2_image(in_path2, array=array2, bands=bands, length=length, width=width, mode='write', data_type='FLOAT')

    out_path = str(tmp_path / 'isce_image_2d_out')
    image_math(in_path1, in_path2, out_path, 'a + b')
    image_obj_out, arrayout = load_isce2_image(out_path)
    assert np.array_equal(array1 + array2, arrayout)


def test_read_product_metadata():
    data_dir = Path('tests/data/test_case')
    product_name = Path('S1_136232_IW2_20200604_20200616_VV_INT80_663F')
    file = Path('S1_136232_IW2_20200604_20200616_VV_INT80_663F.txt')
    metafile = str(data_dir / product_name / file)
    metas = read_product_metadata(metafile)
    assert metas['ReferenceGranule'] == 'S1_136232_IW2_20200604T022315_VV_7C85-BURST'
    assert metas['SecondaryGranule'] == 'S1_136232_IW2_20200616T022316_VV_5D11-BURST'
    assert float(metas['Baseline']) == -66.10716474087386
    assert int(metas['ReferenceOrbitNumber']) == 32861
    assert int(metas['SecondaryOrbitNumber']) == 33036
