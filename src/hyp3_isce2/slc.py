import json
import os
from pathlib import Path
from subprocess import PIPE, run
from zipfile import ZipFile

from hyp3lib.fetch import download_file
from hyp3lib.scene import get_download_url
from shapely import geometry
from shapely.geometry.polygon import Polygon


def get_granule(granule):
    download_url = get_download_url(granule)
    zip_file = download_file(download_url, chunk_size=10485760)
    safe_dir = unzip_granule(zip_file, remove=True)
    return Path.cwd() / safe_dir


def unzip_granule(zip_file, remove=False):
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


def get_dem_bounds(reference_granule: Path, secondary_granule: Path):
    bboxs = []
    for granule in (reference_granule, secondary_granule):
        footprint = get_geometry_from_kml(str(granule / 'preview' / 'map-overlay.kml'))
        bbox = geometry.box(*footprint.bounds)
        bboxs.append(bbox)

    intersection = bboxs[0].intersection(bboxs[1])
    return intersection.bounds


if __name__ == '__main__':
    ref_granule = 'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85'
    sec_granule = 'S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11'
    ref_dir = get_granule(ref_granule)
    sec_dir = get_granule(sec_granule)

    bounds = get_dem_bounds(Path.cwd() / f'{ref_granule}.SAFE', Path.cwd() / f'{sec_granule}.SAFE')
    print('done')
