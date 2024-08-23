"""Create a Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow from a set of bursts"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from shutil import copyfile, make_archive
from typing import Iterable, Optional

import isce  # noqa
from burst2safe.burst2safe import burst2safe
from hyp3lib.util import string_is_true
from isceobj.TopsProc.runMergeBursts import multilook
from osgeo import gdal
from s1_orbits import fetch_for_scene

from hyp3_isce2 import packaging, topsapp
from hyp3_isce2.burst import (
    download_bursts,
    get_burst_params,
    get_isce2_burst_bbox,
    get_region_of_interest,
    multilook_radar_merge_inputs,
    validate_bursts,
)
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.insar_tops import insar_tops_packaged
from hyp3_isce2.logger import configure_root_logger
from hyp3_isce2.s1_auxcal import download_aux_cal
from hyp3_isce2.utils import image_math, isce2_copy, make_browse_image, resample_to_radar_io
from hyp3_isce2.water_mask import create_water_mask


gdal.UseExceptions()

log = logging.getLogger(__name__)


def insar_tops_burst(
    reference_scene: str,
    secondary_scene: str,
    swath_number: int,
    azimuth_looks: int = 4,
    range_looks: int = 20,
    apply_water_mask: bool = False,
) -> Path:
    """Create a burst interferogram

    Args:
        reference_scene: Reference burst name
        secondary_scene: Secondary burst name
        swath_number: Number of swath to grab bursts from (1, 2, or 3) for IW
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks
        apply_water_mask: Whether to apply a pre-unwrap water mask

    Returns:
        Path to results directory
    """
    orbit_dir = Path('orbits')
    aux_cal_dir = Path('aux_cal')
    dem_dir = Path('dem')

    ref_params = get_burst_params(reference_scene)
    sec_params = get_burst_params(secondary_scene)

    ref_metadata, sec_metadata = download_bursts([ref_params, sec_params])

    is_ascending = ref_metadata.orbit_direction == 'ascending'
    ref_footprint = get_isce2_burst_bbox(ref_params)
    sec_footprint = get_isce2_burst_bbox(sec_params)

    insar_roi = get_region_of_interest(ref_footprint, sec_footprint, is_ascending=is_ascending)
    dem_roi = ref_footprint.intersection(sec_footprint).bounds

    if abs(dem_roi[0] - dem_roi[2]) > 180.0 and dem_roi[0] * dem_roi[2] < 0.0:
        raise ValueError('Products that cross the anti-meridian are not currently supported.')

    log.info(f'InSAR ROI: {insar_roi}')
    log.info(f'DEM ROI: {dem_roi}')

    dem_path = download_dem_for_isce2(dem_roi, dem_name='glo_30', dem_dir=dem_dir, buffer=0, resample_20m=False)
    download_aux_cal(aux_cal_dir)

    if range_looks == 5:
        geocode_dem_path = download_dem_for_isce2(
            dem_roi, dem_name='glo_30', dem_dir=dem_dir, buffer=0, resample_20m=True
        )
    else:
        geocode_dem_path = dem_path

    orbit_dir.mkdir(exist_ok=True, parents=True)
    for granule in (ref_params.granule, sec_params.granule):
        log.info(f'Downloading orbit file for {granule}')
        orbit_file = fetch_for_scene(granule, dir=orbit_dir)
        log.info(f'Got orbit file {orbit_file} from s1_orbits')

    config = topsapp.TopsappBurstConfig(
        reference_safe=f'{ref_params.granule}.SAFE',
        secondary_safe=f'{sec_params.granule}.SAFE',
        polarization=ref_params.polarization,
        orbit_directory=str(orbit_dir),
        aux_cal_directory=str(aux_cal_dir),
        roi=insar_roi,
        dem_filename=str(dem_path),
        geocode_dem_filename=str(geocode_dem_path),
        swaths=swath_number,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )
    config_path = config.write_template('topsApp.xml')

    topsapp.run_topsapp_burst(start='startup', end='preprocess', config_xml=config_path)
    topsapp.swap_burst_vrts()
    if apply_water_mask:
        topsapp.run_topsapp_burst(start='computeBaselines', end='filter', config_xml=config_path)
        water_mask_path = 'water_mask.wgs84'
        create_water_mask(str(dem_path), water_mask_path)
        multilook('merged/lon.rdr.full', outname='merged/lon.rdr', alks=azimuth_looks, rlks=range_looks)
        multilook('merged/lat.rdr.full', outname='merged/lat.rdr', alks=azimuth_looks, rlks=range_looks)
        resample_to_radar_io(water_mask_path, 'merged/lat.rdr', 'merged/lon.rdr', 'merged/water_mask.rdr')
        isce2_copy('merged/phsig.cor', 'merged/unmasked.phsig.cor')
        image_math('merged/unmasked.phsig.cor', 'merged/water_mask.rdr', 'merged/phsig.cor', 'a*b')
        topsapp.run_topsapp_burst(start='unwrap', end='unwrap2stage', config_xml=config_path)
        isce2_copy('merged/unmasked.phsig.cor', 'merged/phsig.cor')
    else:
        topsapp.run_topsapp_burst(start='computeBaselines', end='unwrap2stage', config_xml=config_path)
    copyfile('merged/z.rdr.full.xml', 'merged/z.rdr.full.vrt.xml')
    topsapp.run_topsapp_burst(start='geocode', end='geocode', config_xml=config_path)

    return Path('merged')


def insar_tops_single_burst(
    reference: str,
    secondary: str,
    looks: str = '20x4',
    apply_water_mask=False,
    bucket: Optional[str] = None,
    bucket_prefix: str = '',
):
    validate_bursts(reference, secondary)
    swath_number = int(reference[12])
    range_looks, azimuth_looks = [int(value) for value in looks.split('x')]

    log.info('Begin ISCE2 TopsApp run')

    insar_tops_burst(
        reference_scene=reference,
        secondary_scene=secondary,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        swath_number=swath_number,
        apply_water_mask=apply_water_mask,
    )

    log.info('ISCE2 TopsApp run completed successfully')

    multilook_position = multilook_radar_merge_inputs(swath_number, rg_looks=range_looks, az_looks=azimuth_looks)

    pixel_size = packaging.get_pixel_size(looks)
    product_name = packaging.get_product_name(reference, secondary, pixel_spacing=int(pixel_size), slc=False)

    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    packaging.translate_outputs(product_name, pixel_size=pixel_size, include_radar=True, use_multilooked=True)

    unwrapped_phase = f'{product_name}/{product_name}_unw_phase.tif'
    if apply_water_mask:
        packaging.water_mask(unwrapped_phase, f'{product_name}/{product_name}_water_mask.tif')

    make_browse_image(unwrapped_phase, f'{product_name}/{product_name}_unw_phase.png')

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
        multilook_position=multilook_position,
        apply_water_mask=apply_water_mask,
    )
    output_zip = make_archive(base_name=product_name, format='zip', base_dir=product_name)

    if bucket:
        packaging.upload_product_to_s3(product_dir, output_zip, bucket, bucket_prefix)


def insar_tops_multi_burst(
    reference: Iterable[str],
    secondary: Iterable[str],
    swaths: list = [1, 2, 3],
    looks: str = '20x4',
    apply_water_mask=False,
    bucket: Optional[str] = None,
    bucket_prefix: str = '',
):
    validate_bursts(reference, secondary)
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
        bucket_prefix=bucket_prefix,
    )
    log.info('ISCE2 TopsApp run completed successfully')


def oldest_granule_first(g1, g2):
    if g1[14:29] <= g2[14:29]:
        return [g1], [g2]
    return [g2], [g1]


def main():
    """HyP3 entrypoint for the burst TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('granules', type=str.split, nargs='*', help='Reference and secondary scene names')
    parser.add_argument('--reference', type=str.split, nargs='+', help='List of reference scenes"')
    parser.add_argument('--secondary', type=str.split, nargs='+', help='List of secondary scenes"')
    parser.add_argument(
        '--looks', choices=['20x4', '10x2', '5x1'], default='20x4', help='Number of looks to take in range and azimuth'
    )
    parser.add_argument(
        '--apply-water-mask', type=string_is_true, default=False, help='Apply a water body mask before unwrapping.'
    )
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')

    args = parser.parse_args()

    has_granules = args.granules is not None and len(args.granules) > 0
    has_ref_sec = args.reference is not None and args.secondary is not None
    if has_granules and has_ref_sec:
        parser.error('Provide either --reference and --secondary, or the positional granules argument, not both.')
    elif not has_granules and not has_ref_sec:
        parser.error('Either --reference and --secondary, or the positional granules argument, must be provided.')
    elif has_granules:
        warnings.warn(
            'The positional argument for granules is deprecated. Please use --reference and --secondary.',
            DeprecationWarning,
        )
        granules = [item for sublist in args.granules for item in sublist]
        if len(granules) != 2:
            parser.error('No more than two granules may be provided.')
        reference, secondary = oldest_granule_first(granules[0], granules[1])
    else:
        reference = [item for sublist in args.reference for item in sublist]
        secondary = [item for sublist in args.secondary for item in sublist]

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    if len(reference) == 1:
        insar_tops_single_burst(
            reference=reference[0],
            secondary=secondary[0],
            looks=args.looks,
            apply_water_mask=args.apply_water_mask,
            bucket=args.bucket,
            bucket_prefix=args.bucket_prefix,
        )
    else:
        insar_tops_multi_burst(
            reference=reference,
            secondary=secondary,
            looks=args.looks,
            apply_water_mask=args.apply_water_mask,
            bucket=args.bucket,
            bucket_prefix=args.bucket_prefix,
        )
