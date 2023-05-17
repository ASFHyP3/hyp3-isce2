from hyp3_isce2.utils import utm_from_lon_lat, extent_from_geotransform


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
