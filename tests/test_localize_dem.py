from hyp3_isce2 import localize_dem


def test_buffer_extent():
    extent1 = [1, 2, 3, 4]
    assert localize_dem.buffer_extent(extent1, 0) == extent1
    assert localize_dem.buffer_extent(extent1, 0.1) == [0, 1, 4, 5]

    extent2 = [-169.7, 53.3, -167.3, 54.7]
    assert localize_dem.buffer_extent(extent2, 0) == [-170, 53, -167, 55]
    assert localize_dem.buffer_extent(extent2, 0.1) == [-170, 53, -167, 55]
    assert localize_dem.buffer_extent(extent2, 0.3) == [-170, 53, -167, 55]
    assert localize_dem.buffer_extent(extent2, 0.4) == [-171, 52, -166, 56]
