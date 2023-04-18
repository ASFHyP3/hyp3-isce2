"""
ISCE2 stripmap processing
"""

import argparse
import logging
import os
import site
import sys
from pathlib import Path

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.image import create_thumbnail


log = logging.getLogger(__name__)

# ISCE needs its applications to be on the system path.
# See https://github.com/isce-framework/isce2#setup-your-environment
ISCE_APPLICATIONS = Path(site.getsitepackages()[0]) / 'isce' / 'applications'
if str(ISCE_APPLICATIONS) not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] = str(ISCE_APPLICATIONS) + os.pathsep + os.environ['PATH']


def insar_stripmap(reference_scene: str, secondary_scene: str) -> Path:
    """Create an interferogram

    This is a placeholder function. It will be replaced with your actual scientific workflow.

    Args:
        reference_scene: Reference scene name
        secondary_scene: Secondary scene name

    Returns:
        Path to the output files
    """

    raise NotImplementedError('This is a placeholder function. Replace it with your actual scientific workflow.')

    product_file = Path("product_file_name.zip")
    return product_file


def main():
    """ Entrypoint for the stripmap workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', type=str, default='', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', type=str, default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--reference-scene', type=str, required=True)
    parser.add_argument('--secondary-scene', type=str, required=True)

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    log.debug(' '.join(sys.argv))

    product_file = insar_stripmap(
        reference_scene=args.reference_scene,
        secondary_scene=args.secondary_scene,
    )

    log.info('InSAR Stripmap run completed successfully')

    if args.bucket:
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
        browse_images = product_file.with_suffix('.png')
        for browse in browse_images:
            thumbnail = create_thumbnail(browse)
            upload_file_to_s3(browse, args.bucket, args.bucket_prefix)
            upload_file_to_s3(thumbnail, args.bucket, args.bucket_prefix)
