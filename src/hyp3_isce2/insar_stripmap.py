"""
ISCE2 stripmap processing
"""

import argparse
import logging
import site
import sys
from pathlib import Path

from hyp3_isce2 import stripmapapp_alos as stripmapapp

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.image import create_thumbnail
from hyp3_isce2.dem import download_dem_for_isce2
import zipfile
import glob
import os

import asf_search
from shapely.geometry.polygon import Polygon

from hyp3_isce2.logging import configure_root_logger


log = logging.getLogger(__name__)

# ISCE needs its applications to be on the system path.
# See https://github.com/isce-framework/isce2#setup-your-environment
ISCE_APPLICATIONS = Path(site.getsitepackages()[0]) / 'isce' / 'applications'
if str(ISCE_APPLICATIONS) not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] = str(ISCE_APPLICATIONS) + os.pathsep + os.environ['PATH']


# TODO update docstring description
def insar_stripmap(user: str, password: str, reference_scene: str, secondary_scene: str) -> Path:
    """Create an interferogram

    This is a placeholder function. It will be replaced with your actual scientific workflow.

    Args:
        user: Earthdata username
        password: Earthdata password
        reference_scene: Reference scene name
        secondary_scene: Secondary scene name

    Returns:
        Path to the output files
    """
    session = asf_search.ASFSession().auth_with_creds(user, password)

    results = asf_search.search(
        granule_list=[reference_scene, secondary_scene],
        processingLevel=asf_search.L1_0,
    )
    assert len(results) == 2

    polys = []
    durls = []
    for result in results:
        polys.append(Polygon(result.geometry['coordinates'][0]))
        durls.append(result.properties['url'])

    for i in range(len(polys)):
        if i == 0:
            intersection = polys[i].intersection(polys[i + 1])
        else:
            intersection = polys[i].intersection(intersection)

    dem_dir = Path('dem')
    dem_path = download_dem_for_isce2(intersection.bounds, dem_name='glo_30', dem_dir=dem_dir, buffer=0)

    insar_roi = intersection.bounds
    asf_search.download_urls(urls=durls, path='./', session=session, processes=2)

    zips = glob.glob('*.zip')
    for i, zipf in enumerate(sorted(zips[0:2])):
        with zipfile.ZipFile(zipf, 'r') as zip_ref:
            zip_ref.extractall('./')

        if i == 0:
            reference_image = glob.glob('./' + zipf.split('.zip')[0] + '/IMG-*')[0]
            reference_leader = glob.glob('./' + zipf.split('.zip')[0] + '/LED-*')[0]
        else:
            secondary_image = glob.glob('./' + zipf.split('.zip')[0] + '/IMG-*')[0]
            secondary_leader = glob.glob('./' + zipf.split('.zip')[0] + '/LED-*')[0]

        os.remove(zipf)

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
    #stripmapapp.run_stripmapapp(start='rubber_sheet_range', end='geocode', config_xml=config_path)
    #stripmapapp.run_stripmapapp(start='startup', end='geocode', config_xml=config_path)
    #raise NotImplementedError('This is a placeholder function. Replace it with your actual scientific workflow.')

    #product_file = Path("product_file_name.zip")
    return Path('interferogram')


def main():
    """ Entrypoint for the stripmap workflow"""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--username', type=str, required=True)
    parser.add_argument('--password', type=str, required=True)
    parser.add_argument('--reference-scene', type=str, required=True)
    parser.add_argument('--secondary-scene', type=str, required=True)

    args = parser.parse_args()

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    log.info('Begin InSAR Stripmap run')

    product_file = insar_stripmap(
        user=args.username,
        password=args.password,
        reference_scene=args.reference_scene,
        secondary_scene=args.secondary_scene,
    )

    log.info('InSAR Stripmap run completed successfully')

    if args.bucket:
        base_name = f'{args.reference_scene}x{args.secondary_scene}'
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
        browse_images = product_file.with_suffix('.png')
        for browse in browse_images:
            thumbnail = create_thumbnail(browse)
            upload_file_to_s3(browse, args.bucket, args.bucket_prefix)
            upload_file_to_s3(thumbnail, args.bucket, args.bucket_prefix)
