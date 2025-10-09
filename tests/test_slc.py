from pathlib import Path

from shapely.geometry import box

from hyp3_isce2 import slc


def test_dem_bounds(mocker):
    mocker.patch('hyp3_isce2.slc.get_geometry_from_manifest')
    slc.get_geometry_from_manifest.side_effect = [box(-1, -1, 1, 1), box(0, 0, 2, 2)]  # type: ignore[attr-defined]
    bounds = slc.get_dem_bounds(Path('ref'), Path('sec'))
    assert bounds == (0, 0, 1, 1)
