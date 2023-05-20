"""Create a full SLC Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import logging
import os
import site
import sys
from pathlib import Path
from shutil import make_archive

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.get_orb import downloadSentinelOrbitFile
from hyp3lib.image import create_thumbnail

from hyp3_isce2 import slc
from hyp3_isce2 import topsapp
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.logging import configure_root_logger
from hyp3_isce2.s1_auxcal import download_aux_cal


log = logging.getLogger(__name__)

# ISCE needs its applications to be on the system path.
# See https://github.com/isce-framework/isce2#setup-your-environment
ISCE_APPLICATIONS = Path(site.getsitepackages()[0]) / 'isce' / 'applications'
if str(ISCE_APPLICATIONS) not in os.environ['PATH'].split(os.pathsep):
    os.environ['PATH'] = str(ISCE_APPLICATIONS) + os.pathsep + os.environ['PATH']


def insar_tops(
    reference_scene: str,
    secondary_scene: str,
    swaths: list = [1, 2, 3],
    polarization: str = 'VV',
    azimuth_looks: int = 4,
    range_looks: int = 20,
) -> Path:
    """Create a full-SLC interferogram

    Args:
        reference_scene: Reference SLC name
        secondary_scene: Secondary SLC name
        swaths: Swaths to process
        polarization: Polarization to use
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks

    Returns:
        Path to the output files
    """
    orbit_dir = Path('orbits')
    aux_cal_dir = Path('aux_cal')
    dem_dir = Path('dem')

    ref_dir = slc.get_granule(reference_scene)
    sec_dir = slc.get_granule(secondary_scene)
    roi = slc.get_dem_bounds(ref_dir, sec_dir)
    print(f'DEM ROI: {roi}')

    dem_path = download_dem_for_isce2(roi, dem_name='glo_30', dem_dir=dem_dir, buffer=0)
    download_aux_cal(aux_cal_dir)

    orbit_dir.mkdir(exist_ok=True, parents=True)
    for granule in (reference_scene, secondary_scene):
        downloadSentinelOrbitFile(granule, str(orbit_dir))

    config = topsapp.TopsappBurstConfig(
        reference_safe=f'{reference_scene}.SAFE',
        secondary_safe=f'{secondary_scene}.SAFE',
        orbit_directory=str(orbit_dir),
        aux_cal_directory=str(aux_cal_dir),
        dem_filename=str(dem_path),
        roi=roi,
        swaths=swaths,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )
    config_path = config.write_template('topsApp.xml')

    topsapp.run_topsapp_burst(start='startup', end='geocode', config_xml=config_path)

    return Path('merged')


def main():
    """HyP3 entrypoint for the SLC TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--reference-scene', type=str, required=True)
    parser.add_argument('--secondary-scene', type=str, required=True)
    parser.add_argument('--polarization', type=str, default='VV')
    parser.add_argument('--azimuth-looks', type=int, default=4)
    parser.add_argument('--range-looks', type=int, default=20)

    args = parser.parse_args()

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    log.info('Begin ISCE2 TopsApp run')

    product_dir = insar_tops(
        reference_scene=args.reference_scene,
        secondary_scene=args.secondary_scene,
        polarization=args.polarization,
        azimuth_looks=args.azimuth_looks,
        range_looks=args.range_looks,
    )

    log.info('ISCE2 TopsApp run completed successfully')

    if args.bucket:
        base_name = f'{args.reference_scene}x{args.secondary_scene}'
        product_file = make_archive(base_name=base_name, format='zip', base_dir=product_dir)
        upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
        browse_images = product_file.with_suffix('.png')
        for browse in browse_images:
            thumbnail = create_thumbnail(browse)
            upload_file_to_s3(browse, args.bucket, args.bucket_prefix)
            upload_file_to_s3(thumbnail, args.bucket, args.bucket_prefix)
