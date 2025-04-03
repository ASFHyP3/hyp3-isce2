from lxml import etree
from pytest import raises

from hyp3_isce2 import dem


def test_tag_dem_xml_as_ellipsoidal(tmp_path):
    dem_path = tmp_path / 'test_dem.tif'
    xml_path = str(dem_path) + '.xml'
    with open(xml_path, 'w') as f:
        f.write('<root><property name="test"><value>test_value</value></property></root>')

    tagged_xml_path = dem.tag_dem_xml_as_ellipsoidal(dem_path)

    root = etree.parse(tagged_xml_path).getroot()
    assert root.find("./property[@name='reference']/value").text == 'WGS84'
    assert root.find("./property[@name='reference']/doc").text == 'Geodetic datum'


def test_distance_meters_to_degrees():
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=0) == (
        0.000179864321184,
        0.000179864321184,
    )
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=45) == (
        0.000254366562405,
        0.000179864321184,
    )
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=89.9) == (
        0.103054717208573,
        0.000179864321184,
    )
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=-45) == (
        0.000254366562405,
        0.000179864321184,
    )
    assert dem.distance_meters_to_degrees(distance_meters=20, latitude=-89.9) == (
        0.103054717208573,
        0.000179864321184,
    )
    # This is since cos(90) = 0, leading to a divide by zero issue.
    with raises(ZeroDivisionError):
        dem.distance_meters_to_degrees(20, 90)
    with raises(ZeroDivisionError):
        dem.distance_meters_to_degrees(20, -90)
