"""Create a full SLC Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import logging
import sys
from pathlib import Path
from shutil import copyfile, make_archive

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.get_orb import downloadSentinelOrbitFile

from hyp3_isce2 import slc, topsapp
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.logging import configure_root_logger
from hyp3_isce2.s1_auxcal import download_aux_cal


log = logging.getLogger(__name__)


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
    log.info(f'DEM ROI: {roi}')

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

    topsapp.run_topsapp_burst(start='startup', end='unwrap2stage', config_xml=config_path)
    copyfile('merged/z.rdr.full.xml', 'merged/z.rdr.full.vrt.xml')
    topsapp.run_topsapp_burst(start='geocode', end='geocode', config_xml=config_path)

    return Path('merged')


def main():
    """HyP3 entrypoint for the SLC TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--reference-scene', type=str, required=True)
    parser.add_argument('--secondary-scene', type=str, required=True)
    parser.add_argument('--polarization', type=str, default='VV')
    parser.add_argument(
        '--looks',
        choices=['20x4', '10x2', '5x1'],
        default='20x4',
        help='Number of looks to take in range and azimuth'
    )

    args = parser.parse_args()

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    log.info('Begin ISCE2 TopsApp run')

    range_looks, azimuth_looks = [int(looks) for looks in args.looks.split('x')]
    isce_output_dir = insar_tops(
        reference_scene=args.reference_scene,
        secondary_scene=args.secondary_scene,
        polarization=args.polarization,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )

    log.info('ISCE2 TopsApp run completed successfully')

    product_name = f'{args.reference_scene}x{args.secondary_scene}'
    output_zip = make_archive(base_name=product_name, format='zip', base_dir=isce_output_dir)

    if args.bucket:
        upload_file_to_s3(Path(output_zip), args.bucket, args.bucket_prefix)
