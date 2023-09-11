from hyp3_isce2 import insar_tops_burst


def test_get_pixel_size():
    assert insar_tops_burst.get_pixel_size('20x4') == 80.0
    assert insar_tops_burst.get_pixel_size('10x2') == 40.0
    assert insar_tops_burst.get_pixel_size('5x1') == 20.0
