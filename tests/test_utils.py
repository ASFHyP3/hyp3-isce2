import pytest
from osgeo import gdal

from hyp3_isce2.utils import get_utm_proj


@pytest.mark.parametrize('longitude,latitude,epsg', ((-91, 1, 32615), (-91, -1, 32715)))
def test_get_utm_proj(tmp_path, longitude, latitude, epsg):
    # create a temporary gdal dataset
    tmp_ds_path = str(tmp_path / 'tmp.tif')
    tmp_ds = gdal.GetDriverByName('GTiff').Create(tmp_ds_path, 1, 1, 1)
    tmp_ds.SetProjection('EPSG:4326')
    tmp_ds.SetGeoTransform([longitude, 1, 0, latitude, 0, 1])
    del tmp_ds

    utm_proj = get_utm_proj(tmp_ds_path)
    assert utm_proj.to_epsg() == epsg
