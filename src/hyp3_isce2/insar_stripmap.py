"""ISCE2 stripmap processing"""

import argparse
import glob
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from shutil import make_archive

import asf_search
from hyp3lib.aws import upload_file_to_s3
from shapely.geometry.polygon import Polygon

from hyp3_isce2 import stripmapapp_alos as stripmapapp
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.logger import configure_root_logger


log = logging.getLogger(__name__)


def insar_stripmap(reference_scene: str, secondary_scene: str) -> Path:
    """Create a Stripmap interferogram

    Args:
        user: Earthdata username
        password: Earthdata password
        reference_scene: Reference scene name
        secondary_scene: Secondary scene name

    Returns:
        Path to the output files
    """
    scenes = sorted([reference_scene, secondary_scene])
    print(scenes)
    reference_scene = scenes[0]
    secondary_scene = scenes[1]
    products = asf_search.search(
        granule_list=[reference_scene, secondary_scene],
        processingLevel='L1.0',
    )

    if products[0].properties['sceneName'] == reference_scene:
        reference_product = products[0]
        secondary_product = products[1]
    else:
        reference_product = products[1]
        secondary_product = products[0]

    assert reference_product.properties['sceneName'] == reference_scene
    assert secondary_product.properties['sceneName'] == secondary_scene
    products = (reference_product, secondary_product)

    polygons = [Polygon(product.geometry['coordinates'][0]) for product in products]
    insar_roi = polygons[0].intersection(polygons[1]).bounds

    dem_dir = Path('dem')
    dem_dir.mkdir(parents=True, exist_ok=True)
    dem_path = dem_dir / 'full_res.dem.wgs84'
    download_dem_for_isce2(insar_roi, dem_path, pixel_size=30.0)

    urls = [product.properties['url'] for product in products]
    asf_search.download_urls(urls=urls, path=os.getcwd(), processes=2)

    zip_paths = [product.properties['fileName'] for product in products]
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall()
        os.remove(zip_path)

    reference_image = get_product_file(reference_product, 'IMG-')
    reference_leader = get_product_file(reference_product, 'LED-')

    secondary_image = get_product_file(secondary_product, 'IMG-')
    secondary_leader = get_product_file(secondary_product, 'LED-')

    config = stripmapapp.StripmapappConfig(
        reference_image=reference_image,
        reference_leader=reference_leader,
        secondary_image=secondary_image,
        secondary_leader=secondary_leader,
        roi=insar_roi,
        dem_filename=str(dem_path),
    )
    config_path = config.write_template('stripmapApp.xml')

    stripmapapp.run_stripmapapp(start='startup', end='geocode', config_xml=config_path)

    product_dir = Path(f'{reference_scene}x{secondary_scene}')
    (product_dir / 'interferogram').mkdir(parents=True)

    for filename in os.listdir('interferogram'):
        path = Path('interferogram') / filename
        if os.path.isfile(path):
            shutil.move(path, product_dir / path)

    shutil.move('geometry', product_dir)
    shutil.move('ionosphere', product_dir)

    return product_dir


def get_product_file(product: asf_search.ASFProduct, file_prefix: str) -> str:
    paths = glob.glob(str(Path(product.properties['fileID']) / f'{file_prefix}*'))
    assert len(paths) > 0
    return paths[0]


def main():
    """Entrypoint for the stripmap workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--reference', type=str, required=True)
    parser.add_argument('--secondary', type=str, required=True)

    args = parser.parse_args()

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    log.info('Begin InSAR Stripmap run')

    product_dir = insar_stripmap(
        reference_scene=args.reference,
        secondary_scene=args.secondary,
    )

    log.info('InSAR Stripmap run completed successfully')

    output_zip = make_archive(base_name=product_dir.name, format='zip', base_dir=product_dir)

    if args.bucket:
        # TODO do we want browse images?

        upload_file_to_s3(Path(output_zip), args.bucket, args.bucket_prefix)

        # TODO upload individual files to S3?
