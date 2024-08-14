import json
import os
from pathlib import Path
from subprocess import PIPE, run
from zipfile import ZipFile

import lxml.etree as ET
from hyp3lib.fetch import download_file
from hyp3lib.scene import get_download_url
from shapely import geometry
from shapely.geometry.polygon import Polygon


def get_granule(granule: str) -> Path:
    """Download and unzip a Sentinel-1 SLC granule

    Args:
        granule: The granule name with no extension

    Returns:
        The path to the unzipped granule
    """
    if Path(f'{granule}.SAFE').exists():
        print('SAFE file already exists, skipping download.')
        return Path.cwd() / f'{granule}.SAFE'

    download_url = get_download_url(granule)
    zip_file = download_file(download_url, chunk_size=10485760)
    safe_dir = unzip_granule(zip_file, remove=True)
    return Path.cwd() / safe_dir


def unzip_granule(zip_file: Path, remove: bool = False) -> Path:
    with ZipFile(zip_file) as z:
        z.extractall()
        safe_dir = next(item.filename for item in z.infolist() if item.is_dir() and item.filename.endswith('.SAFE/'))
    if remove:
        os.remove(zip_file)
    return safe_dir.strip('/')


def get_geometry_from_kml(kml_file: str) -> Polygon:
    cmd = f'ogr2ogr -wrapdateline -datelineoffset 20 -f GeoJSON -mapfieldtype DateTime=String /vsistdout {kml_file}'
    geojson_str = run(cmd.split(' '), stdout=PIPE, check=True).stdout
    geojson = json.loads(geojson_str)['features'][0]['geometry']
    return geometry.shape(geojson)


def get_geometry_from_manifest(manifest_path: Path):
    manifest = ET.parse(manifest_path).getroot()
    frame_element = [x for x in manifest.findall('.//metadataObject') if x.get('ID') == 'measurementFrameSet'][0]
    frame_string = frame_element.find('.//{http://www.opengis.net/gml}coordinates').text
    coord_strings = [pair.split(',') for pair in frame_string.split(' ')]
    coords = [(float(lon), float(lat)) for lat, lon in coord_strings]
    footprint = Polygon(coords)
    return footprint


def get_relative_orbit(granule: Path):
    manifest_path = granule / 'manifest.safe'
    manifest = ET.parse(manifest_path).getroot()
    
    frame_element = [x for x in manifest.findall('.//metadataObject') if x.get('ID') == 'measurementOrbitReference'][0]
    frame_element = [x for x in frame_element.findall('.//metadataWrap/xmlData')][0]
    
    print(frame_element.get('type'))
    relative_orbit = frame_element.findall('.//relativeOrbitNumber')[0].text
    return relative_orbit


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
