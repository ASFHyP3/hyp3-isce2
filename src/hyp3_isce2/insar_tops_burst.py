"""
ISCE2 processing
"""

import argparse
import logging
import os
import site
import sys
from pathlib import Path
from shutil import make_archive
from typing import Literal

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.get_orb import downloadSentinelOrbitFile
from hyp3lib.image import create_thumbnail

from hyp3_isce2 import topsapp
from hyp3_isce2.burst import BurstParams, download_bursts, get_region_of_interest
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.s1_auxcal import download_aux_cal


log = logging.getLogger(__name__)

# ISCE needs its applications to be on the system path.
# See https://github.com/isce-framework/isce2#setup-your-environment
ISCE_APPLICATIONS = Path(site.getsitepackages()[0]) / 'isce' / 'applications'
if str(ISCE_APPLICATIONS) not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] = str(ISCE_APPLICATIONS) + os.pathsep + os.environ['PATH']


def insar_tops_burst(
    reference_scene: str,
    secondary_scene: str,
    swath_number: int,
    reference_burst_number: int,
    secondary_burst_number: int,
    polarization: str = 'VV',
    azimuth_looks: int = 4,
    range_looks: int = 20,
) -> Path:
    """Create a burst interferogram

    Args:
        reference_scene: Reference SLC name
        secondary_scene: Secondary SLC name
        swath_number: Number of swath to grab bursts from (1, 2, or 3) for IW
        reference_burst_number: Number of burst to download for reference (0-indexed from first collect)
        secondary_burst_number: Number of burst to download for secondary (0-indexed from first collect)
        polarization: Polarization to use
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks

    Returns:
        Path to the output files
    """
    orbit_dir = Path('orbits')
    aux_cal_dir = Path('aux_cal')
    dem_dir = Path('dem')
    ref_params = BurstParams(reference_scene, f'IW{swath_number}', polarization.upper(), reference_burst_number)
    sec_params = BurstParams(secondary_scene, f'IW{swath_number}', polarization.upper(), secondary_burst_number)
    ref_metadata, sec_metadata = download_bursts([ref_params, sec_params])

    is_ascending = ref_metadata.orbit_direction == 'ascending'
    insar_roi = get_region_of_interest(ref_metadata.footprint, sec_metadata.footprint, is_ascending=is_ascending)
    dem_roi = ref_metadata.footprint.intersection(sec_metadata.footprint).bounds
    print(f'InSAR ROI: {insar_roi}')
    print(f'DEM ROI: {dem_roi}')

    dem_path = download_dem_for_isce2(dem_roi, dem_name='glo_30', dem_dir=dem_dir, buffer=0)
    download_aux_cal(aux_cal_dir)

    orbit_dir.mkdir(exist_ok=True, parents=True)
    for granule in (ref_params.granule, sec_params.granule):
        downloadSentinelOrbitFile(granule, str(orbit_dir))

    config = topsapp.TopsappBurstConfig(
        reference_safe=f'{ref_params.granule}.SAFE',
        secondary_safe=f'{sec_params.granule}.SAFE',
        orbit_directory=str(orbit_dir),
        aux_cal_directory=str(aux_cal_dir),
        roi=insar_roi,
        dem_filename=str(dem_path),
        swath=swath_number,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )
    config_path = config.write_template('topsApp.xml')

    topsapp.run_topsapp_burst(start='startup', end='preprocess', config_xml=config_path)
    topsapp.swap_burst_vrts()
    topsapp.run_topsapp_burst(start='computeBaselines', end='geocode', config_xml=config_path)

    return Path('merged')


def run_insar_tops_burst(args):
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))

    product_dir = insar_tops_burst(
        reference_scene=args.reference_scene,
        secondary_scene=args.secondary_scene,
        swath_number=args.swath_number,
        polarization=args.polarization,
        reference_burst_number=args.reference_burst_number,
        secondary_burst_number=args.secondary_burst_number,
        azimuth_looks=args.azimuth_looks,
        range_looks=args.range_looks,
    )

    log.info('ISCE2 TopsApp run completed successfully')
    return product_dir


def _get_cli(interface: Literal['hyp3', 'main']) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    if interface == 'hyp3':
        parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
        parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    elif interface == 'main':
        pass
    else:
        raise NotImplementedError(f'Unknown interface: {interface}')

    parser.add_argument('--reference-scene', type=str, required=True)
    parser.add_argument('--secondary-scene', type=str, required=True)
    parser.add_argument('--swath-number', type=int, required=True)
    parser.add_argument('--polarization', type=str, default='VV')
    parser.add_argument('--reference-burst-number', type=int, required=True)
    parser.add_argument('--secondary-burst-number', type=int, required=True)
    parser.add_argument('--azimuth-looks', type=int, default=4)
    parser.add_argument('--range-looks', type=int, default=20)

    return parser


def hyp3():
    """ HyP3 entrypoint for the burst workflow

    Uses the HyP3 specific arguments and uploads the product to S3
    """
    parser = _get_cli(interface='hyp3')
    args = parser.parse_args()

    product_dir = run_insar_tops_burst(args)

    if args.bucket:
        reference_name = (
            f'{args.reference_scene}_IW{args.swath_number}_{args.polarization}_{args.reference_burst_number}'
        )
        secondary_name = (
            f'{args.secondary_scene}_IW{args.swath_number}_{args.polarization}_{args.secondary_burst_number}'
        )
        base_name = f'{reference_name}x{secondary_name}'
        product_file = make_archive(base_name=base_name, format='zip', base_dir=product_dir)
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
        browse_images = product_file.with_suffix('.png')
        for browse in browse_images:
            thumbnail = create_thumbnail(browse)
            upload_file_to_s3(browse, args.bucket, args.bucket_prefix)
            upload_file_to_s3(thumbnail, args.bucket, args.bucket_prefix)


def main():
    """Main entrypoint for the burst workflow"""
    parser = _get_cli(interface='main')
    args = parser.parse_args()

    run_insar_tops_burst(args)
