"""Create a multi-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import logging
import sys
from typing import Iterable, Optional

import isce  # noqa
from burst2safe.burst2safe import burst2safe
from hyp3lib.util import string_is_true
from osgeo import gdal

from hyp3_isce2.insar_tops import insar_tops_packaged
from hyp3_isce2.insar_tops_burst import insar_tops_single_burst
from hyp3_isce2.logger import configure_root_logger


gdal.UseExceptions()

log = logging.getLogger(__name__)


def insar_tops_multi_burst(
    reference: Iterable[str],
    secondary: Iterable[str],
    swaths: list = [1, 2, 3],
    looks: str = '20x4',
    apply_water_mask=False,
    bucket: Optional[str] = None,
    bucket_prefix: str = '',
):
    ref_ids = [g.split('_')[1] + '_' + g.split('_')[2] + '_' + g.split('_')[4] for g in reference]
    sec_ids = [g.split('_')[1] + '_' + g.split('_')[2] + '_' + g.split('_')[4] for g in secondary]

    if ref_ids != sec_ids:
        raise Exception('The reference bursts and secondary bursts do not match')

    reference_safe_path = burst2safe(reference)
    reference_safe = reference_safe_path.name.split('.')[0]
    secondary_safe_path = burst2safe(secondary)
    secondary_safe = secondary_safe_path.name.split('.')[0]

    range_looks, azimuth_looks = [int(value) for value in looks.split('x')]
    swaths = list(set(int(granule.split('_')[2][2]) for granule in reference))
    polarization = reference[0].split('_')[4]

    log.info('Begin ISCE2 TopsApp run')
    insar_tops_packaged(
        reference=reference_safe,
        secondary=secondary_safe,
        swaths=swaths,
        polarization=polarization,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        apply_water_mask=apply_water_mask,
        bucket=bucket,
        bucket_prefix=bucket_prefix
    )
    log.info('ISCE2 TopsApp run completed successfully')


def main():
    """HyP3 entrypoint for the burst TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument(
        '--looks', choices=['20x4', '10x2', '5x1'], default='20x4', help='Number of looks to take in range and azimuth'
    )
    parser.add_argument(
        '--apply-water-mask',
        type=string_is_true,
        default=False,
        help='Apply a water body mask before unwrapping.',
    )
    parser.add_argument(
        '--reference',
        type=str.split,
        nargs='+',
        help='List of reference scenes"'
    )
    parser.add_argument(
        '--secondary',
        type=str.split,
        nargs='+',
        help='List of secondary scenes"'
    )

    args = parser.parse_args()

    references = [item for sublist in args.reference for item in sublist]
    secondaries = [item for sublist in args.secondary for item in sublist]

    if len(references) < 1 or len(secondaries) < 1:
        parser.error("Must include at least 1 reference and 1 secondary")
    if (len(references) != len(secondaries)):
        parser.error("Must have the same number of references and secondaries")

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    if len(references) == 1:
        insar_tops_single_burst(
            reference=references[0],
            secondary=secondaries[0],
            looks=args.looks,
            apply_water_mask=args.apply_water_mask,
            bucket=args.bucket,
            bucket_prefix=args.bucket_prefix,
        )
    else:
        insar_tops_multi_burst(
            reference=references,
            secondary=secondaries,
            looks=args.looks,
            apply_water_mask=args.apply_water_mask,
            bucket=args.bucket,
            bucket_prefix=args.bucket_prefix,
        )
