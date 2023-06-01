from osgeo import gdal

from hyp3_isce2.utils import GDALConfigManager, earlier_granule_first, extent_from_geotransform, utm_from_lon_lat

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


def test_earlier_granule_first():
    latest = "S1_249434_IW1_20230511T170732_VV_07DE-BURST"
    oldest = "S1_249434_IW1_20230523T170733_VV_8850-BURST"
    assert earlier_granule_first(oldest, latest) == (oldest, latest)
    assert earlier_granule_first(latest, oldest) == (oldest, latest)