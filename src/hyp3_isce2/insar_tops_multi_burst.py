"""Create a Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow from a set of bursts"""

import argparse
import logging
import sys
from pathlib import Path
from shutil import make_archive

import isce  # noqa: F401
from burst2safe.burst2safe import burst2safe
from hyp3lib.util import string_is_true
from osgeo import gdal

from hyp3_isce2 import packaging
from hyp3_isce2.burst import (
    validate_bursts,
)
from hyp3_isce2.insar_tops import insar_tops
from hyp3_isce2.logger import configure_root_logger
from hyp3_isce2.utils import make_browse_image


gdal.UseExceptions()

log = logging.getLogger(__name__)


def insar_tops_multi_burst(
    reference_bursts: list[str],
    secondary_bursts: list[str],
    range_looks: int = 20,
    azimuth_looks: int = 4,
    apply_water_mask=False,
) -> tuple[Path, Path]:
    validate_bursts(reference_bursts, secondary_bursts)

    reference_safe_path = burst2safe(reference_bursts)
    secondary_safe_path = burst2safe(secondary_bursts)

    swaths = list(set(int(granule.split('_')[2][2]) for granule in reference_bursts))
    polarization = reference_bursts[0].split('_')[4]

    insar_tops(
        reference_safe_dir=reference_safe_path,
        secondary_safe_dir=secondary_safe_path,
        swaths=swaths,
        polarization=polarization,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        apply_water_mask=apply_water_mask,
    )

    pixel_size = packaging.get_pixel_size(f'{range_looks}x{azimuth_looks}')
    product_name = packaging.get_product_name(
        reference_bursts,
        secondary_bursts,
        relative_orbit=0,  # TODO
        pixel_spacing=int(pixel_size),
        polarization=polarization,
    )

    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    packaging.translate_outputs(product_name, pixel_size=pixel_size)

    unwrapped_phase = f'{product_name}/{product_name}_unw_phase.tif'
    if apply_water_mask:
        packaging.water_mask(unwrapped_phase, f'{product_name}/{product_name}_water_mask.tif')

    make_browse_image(unwrapped_phase, f'{product_name}/{product_name}_unw_phase.png')
    packaging.make_readme(
        product_dir=product_dir,
        product_name=product_name,
        reference_scenes=reference_bursts,
        secondary_scenes=secondary_bursts,
        range_looks=range_looks,
        azimuth_looks=azimuth_looks,
        apply_water_mask=apply_water_mask,
    )
    packaging.make_parameter_file(
        Path(f'{product_name}/{product_name}.txt'),
        reference_scenes=reference_bursts,
        secondary_scenes=secondary_bursts,
        processing_path=Path.cwd(),
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        apply_water_mask=apply_water_mask,
        reference_safe_path=reference_safe_path,
        secondary_safe_path=secondary_safe_path,
    )

    output_zip = make_archive(base_name=product_name, format='zip', base_dir=product_name)

    return product_dir, Path(output_zip)


def main():
    """HyP3 entrypoint for the burst TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--reference', type=str.split, nargs='+', help='List of reference scenes"')
    parser.add_argument('--secondary', type=str.split, nargs='+', help='List of secondary scenes"')
    parser.add_argument(
        '--looks',
        choices=['20x4', '10x2', '5x1'],
        default='20x4',
        help='Number of looks to take in range and azimuth',
    )
    parser.add_argument(
        '--apply-water-mask',
        type=string_is_true,
        default=False,
        help='Apply a water body mask before unwrapping.',
    )
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')

    args = parser.parse_args()

    reference = [item for sublist in args.reference for item in sublist]
    secondary = [item for sublist in args.secondary for item in sublist]
    range_looks, azimuth_looks = args.looks.split('x')

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    product_dir, output_zip = insar_tops_multi_burst(
        reference_bursts=reference,
        secondary_bursts=secondary,
        range_looks=range_looks,
        azimuth_looks=azimuth_looks,
        apply_water_mask=args.apply_water_mask,
    )

    if args.bucket:
        packaging.upload_product_to_s3(product_dir, output_zip, args.bucket, args.bucket_prefix)
