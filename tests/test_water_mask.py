import numpy as np
from osgeo import gdal, osr

from hyp3_isce2 import water_mask

gdal.UseExceptions()


def get_envelope_with_args(out_path, xres, yres, bounds):
    out_img_path = str(out_path)
    driver = gdal.GetDriverByName('GTiff')
    spatref = osr.SpatialReference()
    spatref.ImportFromEPSG(4326)
    wkt = spatref.ExportToWkt()

    xmin, ymin, xmax, ymax = bounds

    xsize = abs(int((xmax - xmin) / xres))
    ysize = abs(int((ymax - ymin) / yres))

    ds = driver.Create(out_img_path, xsize, ysize, options=['COMPRESS=LZW', 'TILED=YES'])
    ds.SetProjection(wkt)
    ds.SetGeoTransform([xmin, xres, 0, ymin, 0, yres])
    ds.GetRasterBand(1).Fill(0)
    ds.FlushCache()
    ds = None

    return water_mask.get_envelope(out_img_path)


def test_get_envelope(tmp_path):
    out_path_1 = str(tmp_path / 'envelope1.tif')
    out_path_2 = str(tmp_path / 'envelope2.tif')
    envelope1, epsg1 = get_envelope_with_args(out_path_1, 0.01, 0.01, [0, 40, 10, 50])
    envelope2, epsg2 = get_envelope_with_args(out_path_2, 0.01, 0.01, [-179, 40, 179, 50])
    assert epsg1 == 4326
    assert epsg2 == 4326
    assert np.all(envelope1.bounds.values == np.asarray([[0.0, 40.0, 10.0, 50.0]]))
    assert np.all(envelope2.bounds.values == np.asarray([[-179.0, 40.0, 179.0, 50.0]]))


def test_create_water_mask_with_no_water(tmp_path, test_data_dir):
    input_tif = str(test_data_dir / 'test_geotiff.tif')
    output_tif = str(tmp_path / 'water_mask.tif')
    water_mask.create_water_mask(input_tif, output_tif)

    info = gdal.Info(output_tif, format='json', stats=True)
    assert info['size'] == [20, 20]
    assert info['geoTransform'] == [440720.0, 60.0, 0.0, 3751320.0, 0.0, -60.0]
    assert info['bands'][0]['type'] == 'Byte'
    assert info['bands'][0]['minimum'] == 1
    assert info['bands'][0]['maximum'] == 1
    assert info['bands'][0]['block'] == [256, 256]
    assert info['metadata']['']['AREA_OR_POINT'] == 'Area'
    assert info['metadata']['IMAGE_STRUCTURE']['COMPRESSION'] == 'LZW'


def test_create_water_mask_with_water_and_land(tmp_path, test_data_dir):
    input_tif = str(test_data_dir / 'water_mask_input.tif')
    output_tif = str(tmp_path / 'water_mask.tif')
    water_mask.create_water_mask(input_tif, output_tif)

    info = gdal.Info(output_tif, format='json')
    assert info['geoTransform'] == [200360.0, 80.0, 0.0, 1756920.0, 0.0, -80.0]
    assert info['bands'][0]['type'] == 'Byte'
    assert info['bands'][0]['block'] == [256, 256]
    assert info['metadata']['']['AREA_OR_POINT'] == 'Point'
    assert info['metadata']['IMAGE_STRUCTURE']['COMPRESSION'] == 'LZW'

    ds = gdal.Open(str(output_tif))
    data = ds.GetRasterBand(1).ReadAsArray()
    expected = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    assert np.array_equal(data, expected)
    del ds
