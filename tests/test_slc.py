from pathlib import Path

from shapely.geometry import box

from hyp3_isce2 import slc


def test_get_geometry_from_kml(test_data_dir):
    kml = test_data_dir / 'alaska.kml'
    expected = {
        'type': 'Polygon',
        'coordinates': (
            (
                (-154.0, 71.0),
                (-147.0, 71.0),
                (-146.0, 70.0),
                (-153.0, 69.0),
                (-154.0, 71.0),
            ),
        ),
    }
    geometry = slc.get_geometry_from_kml(kml)
    assert geometry.__geo_interface__ == expected

    kml = test_data_dir / 'antimeridian.kml'
    expected = {
        'type': 'MultiPolygon',
        'coordinates': [
            (
                (
                    (176.0, 51.0),
                    (177.0, 52.0),
                    (180.0, 52.0),
                    (180.0, 50.2),
                    (176.0, 51.0),
                ),
            ),
            (
                (
                    (-180.0, 50.2),
                    (-180.0, 52.0),
                    (-179.0, 52.0),
                    (-179.0, 50.0),
                    (-180.0, 50.2),
                ),
            ),
        ],
    }
    geometry = slc.get_geometry_from_kml(kml)
    assert geometry.__geo_interface__ == expected


def test_dem_bounds(mocker):
    mocker.patch('hyp3_isce2.slc.get_geometry_from_manifest')
    slc.get_geometry_from_manifest.side_effect = [box(-1, -1, 1, 1), box(0, 0, 2, 2)] # type: ignore
    bounds = slc.get_dem_bounds(Path('ref'), Path('sec'))
    assert bounds == (0, 0, 1, 1)
