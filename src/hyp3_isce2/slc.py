from pathlib import Path

import lxml.etree as ET
from shapely import geometry
from shapely.geometry.polygon import Polygon


def get_geometry_from_manifest(manifest_path: Path):
    manifest = ET.parse(manifest_path).getroot()

    frame_element: ET._Element = [
        x for x in manifest.findall('.//metadataObject') if x.get('ID') == 'measurementFrameSet'
    ][0]
    coord_element = frame_element.find('.//{http://www.opengis.net/gml}coordinates')
    assert isinstance(coord_element, ET._Element)
    assert isinstance(coord_element.text, str)
    frame_string = coord_element.text
    coord_strings = [pair.split(',') for pair in frame_string.split(' ')]
    coords = [(float(lon), float(lat)) for lat, lon in coord_strings]
    footprint = Polygon(coords)
    return footprint


def get_dem_bounds(reference_granule: Path, secondary_granule: Path) -> tuple:
    """Get the bounds of the DEM to use in processing from SAFE KML files

    Args:
        reference_granule: The path to the reference granule
        secondary_granule: The path to the secondary granule

    Returns:
        The bounds of the DEM to use for ISCE2 processing
    """
    bboxs = []
    for granule in (reference_granule, secondary_granule):
        footprint = get_geometry_from_manifest(granule / 'manifest.safe')
        bbox = geometry.box(*footprint.bounds)
        bboxs.append(bbox)

    intersection = bboxs[0].intersection(bboxs[1])
    return intersection.bounds
