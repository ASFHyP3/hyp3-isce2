from re import match

from hyp3_isce2 import packaging


def test_get_product_name():
    reference_name = 'S1_136231_IW2_20200604T022312_VV_7C85-BURST'
    secondary_name = 'S1_136231_IW2_20200616T022313_VV_5D11-BURST'

    name_20m = packaging.get_product_name(reference_name, secondary_name, pixel_spacing=20.0, slc=False)
    name_80m = packaging.get_product_name(reference_name, secondary_name, pixel_spacing=80, slc=False)

    assert match('[A-F0-9]{4}', name_20m[-4:]) is not None
    assert match('[A-F0-9]{4}', name_80m[-4:]) is not None

    assert name_20m.startswith('S1_136231_IW2_20200604_20200616_VV_INT20')
    assert name_80m.startswith('S1_136231_IW2_20200604_20200616_VV_INT80')


def test_get_pixel_size():
    assert packaging.get_pixel_size('20x4') == 80.0
    assert packaging.get_pixel_size('10x2') == 40.0
    assert packaging.get_pixel_size('5x1') == 20.0
