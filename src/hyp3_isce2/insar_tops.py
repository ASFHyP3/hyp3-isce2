"""Create a full SLC Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import logging
import sys
from pathlib import Path
from shutil import copyfile, make_archive

from hyp3lib.util import string_is_true
from s1_orbits import fetch_for_scene

from hyp3_isce2 import packaging, slc, topsapp
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.logger import configure_root_logger
from hyp3_isce2.s1_auxcal import download_aux_cal


log = logging.getLogger(__name__)


def insar_tops(
    reference_scene: str,
    secondary_scene: str,
    swaths: list = [1, 2, 3],
    polarization: str = 'VV',
    azimuth_looks: int = 4,
    range_looks: int = 20,
    download: bool = True,
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

    if download:
        ref_dir = slc.get_granule(reference_scene)
        sec_dir = slc.get_granule(secondary_scene)
    else:
        ref_dir = Path(reference_scene + '.SAFE')
        sec_dir = Path(secondary_scene + '.SAFE')
    roi = slc.get_dem_bounds(ref_dir, sec_dir)
    log.info(f'DEM ROI: {roi}')

    dem_path = download_dem_for_isce2(roi, dem_name='glo_30', dem_dir=dem_dir, buffer=0)
    download_aux_cal(aux_cal_dir)

    orbit_dir.mkdir(exist_ok=True, parents=True)
    for granule in (reference_scene, secondary_scene):
        log.info(f'Downloading orbit file for {granule}')
        orbit_file = fetch_for_scene(granule, dir=orbit_dir)
        log.info(f'Got orbit file {orbit_file} from s1_orbits')

    config = topsapp.TopsappBurstConfig(
        reference_safe=f'{reference_scene}.SAFE',
        secondary_safe=f'{secondary_scene}.SAFE',
        polarization=polarization,
        orbit_directory=str(orbit_dir),
        aux_cal_directory=str(aux_cal_dir),
        dem_filename=str(dem_path),
        geocode_dem_filename=str(dem_path),
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


def insar_tops_packaged(
    reference: str,
    secondary: str,
    swaths: list = [1, 2, 3],
    polarization: str = 'VV',
    azimuth_looks: int = 4,
    range_looks: int = 20,
    apply_water_mask: bool = True,
    download: bool = True,
    bucket: str = None,
    bucket_prefix: str = '',
) -> Path:
    """Create a full-SLC interferogram

    Args:
        reference: Reference SLC name
        secondary: Secondary SLC name
        swaths: Swaths to process
        polarization: Polarization to use
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks
        apply_water_mask: Apply water mask to unwrapped phase
        download: Download the SLCs
        bucket: AWS S3 bucket to upload the final product to
        bucket_prefix: Bucket prefix to prefix to use when uploading the final product

    Returns:
        Path to the output files
    """
    pixel_size = packaging.get_pixel_size(f'{range_looks}x{azimuth_looks}')
    product_name = packaging.get_product_name(reference, secondary, pixel_spacing=int(pixel_size))

    log.info('Begin ISCE2 TopsApp run')
    insar_tops(reference, secondary, download=False)
    log.info('ISCE2 TopsApp run completed successfully')

    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    packaging.translate_outputs(product_name, pixel_size=pixel_size)

    unwrapped_phase = f'{product_name}/{product_name}_unw_phase.tif'
    if apply_water_mask:
        packaging.water_mask(unwrapped_phase, f'{product_name}/{product_name}_water_mask.tif')

    packaging.make_browse_image(unwrapped_phase, f'{product_name}/{product_name}_unw_phase.png')
    packaging.make_readme(
        product_dir=product_dir,
        product_name=product_name,
        reference_scene=reference,
        secondary_scene=secondary,
        range_looks=range_looks,
        azimuth_looks=azimuth_looks,
        apply_water_mask=apply_water_mask,
    )
    packaging.make_parameter_file(
        Path(f'{product_name}/{product_name}.txt'),
        reference_scene=reference,
        secondary_scene=secondary,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        apply_water_mask=apply_water_mask,
    )
    output_zip = make_archive(base_name=product_name, format='zip', base_dir=product_name)
    if bucket:
        packaging.upload_product_to_s3(product_dir, output_zip, bucket, bucket_prefix)


def main():
    """HyP3 entrypoint for the SLC TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--reference', type=str, help='Reference granule')
    parser.add_argument('--secondary', type=str, help='Secondary granule')
    parser.add_argument('--polarization', type=str, defualt='VV', help='Polarization to use')
    parser.add_argument(
        '--looks', choices=['20x4', '10x2', '5x1'], default='20x4', help='Number of looks to take in range and azimuth'
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
    configure_root_logger()
    log.debug(' '.join(sys.argv))

    range_looks, azimuth_looks = [int(looks) for looks in args.looks.split('x')]
    if args.polarization not in ['VV', 'VH', 'HV', 'HH']:
        raise ValueError('Polarization must be one of VV, VH, HV, or HH')

    insar_tops_packaged(
        reference_scene=args.reference,
        secondary_scene=args.secondary,
        polarization=args.polarization,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        apply_water_mask=args.apply_water_mask,
        bucket=args.bucket,
        bucket_prefix=args.bucket_prefix,
    )

    log.info('ISCE2 TopsApp run completed successfully')
