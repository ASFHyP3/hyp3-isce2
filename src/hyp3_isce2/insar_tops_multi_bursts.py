"""A workflow for merging standard burst InSAR products."""
import argparse
import copy
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from shutil import make_archive
from typing import Iterable, Optional

import isce
from burst2safe.burst2safe import burst2safe
from hyp3lib.aws import upload_file_to_s3
from hyp3lib.image import create_thumbnail
from hyp3lib.util import string_is_true
from lxml import etree
from osgeo import gdal, gdalconst

import hyp3_isce2
from hyp3_isce2.insar_tops import insar_tops
from hyp3_isce2.insar_tops_burst import convert_raster_from_isce2_gdal, find_product, get_pixel_size
from hyp3_isce2.utils import ParameterFile, get_projection, make_browse_image, utm_from_lon_lat


log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO, force=True)
log = logging.getLogger(__name__)


@dataclass
class ISCE2Dataset:
    name: str
    suffix: str
    band: Iterable[int]
    dtype: Optional[int] = gdalconst.GDT_Float32


def get_product_name(reference_scene: str, secondary_scene: str, pixel_spacing: int) -> str:
    """Get the name of the interferogram product.

    Args:
        reference_scene: The reference burst name.
        secondary_scene: The secondary burst name.
        pixel_spacing: The spacing of the pixels in the output image.

    Returns:
        The name of the interferogram product.
    """

    reference_split = reference_scene.split('_')
    secondary_split = secondary_scene.split('_')

    platform = reference_split[0][0:2]
    reference_date = reference_split[5][0:8]
    secondary_date = secondary_split[5][0:8]
    product_type = 'INT'
    pixel_spacing = str(int(pixel_spacing))
    product_id = token_hex(2).upper()

    return '_'.join(
        [
            platform,
            reference_date,
            secondary_date,
            product_type + pixel_spacing,
            product_id,
        ]
    )


def translate_outputs(product_name: str, pixel_size: float, include_radar: bool = False) -> None:
    """Translate ISCE outputs to a standard GTiff format with a UTM projection.
    Assume you are in the top level of an ISCE run directory

    Args:
        product_name: Name of the product
        pixel_size: Pixel size
        include_radar: Flag to include the full resolution radar geometry products in the output
    """

    src_ds = gdal.Open('merged/filt_topophase.unw.geo')
    src_geotransform = src_ds.GetGeoTransform()
    src_projection = src_ds.GetProjection()

    target_ds = gdal.Open('merged/dem.crop', gdal.GA_Update)
    target_ds.SetGeoTransform(src_geotransform)
    target_ds.SetProjection(src_projection)

    del src_ds, target_ds

    datasets = [
        ISCE2Dataset('merged/filt_topophase.unw.geo', 'unw_phase', [2]),
        ISCE2Dataset('merged/phsig.cor.geo', 'corr', [1]),
        ISCE2Dataset('merged/dem.crop', 'dem', [1]),
        ISCE2Dataset('merged/filt_topophase.unw.conncomp.geo', 'conncomp', [1]),
    ]

    rdr_datasets = [
        ISCE2Dataset(
            find_product('merged/filt_topophase.flat.vrt'),
            'wrapped_phase_rdr',
            [1],
            gdalconst.GDT_CFloat32,
        ),
        ISCE2Dataset(find_product('merged/lat.rdr.full.vrt'), 'lat_rdr', [1]),
        ISCE2Dataset(find_product('merged/lon.rdr.full.vrt'), 'lon_rdr', [1]),
        ISCE2Dataset(find_product('merged/los.rdr.full.vrt'), 'los_rdr', [1, 2]),
    ]
    if include_radar:
        datasets += rdr_datasets

    for dataset in datasets:
        out_file = str(Path(product_name) / f'{product_name}_{dataset.suffix}.tif')
        gdal.Translate(
            destName=out_file,
            srcDS=dataset.name,
            bandList=dataset.band,
            format='GTiff',
            outputType=dataset.dtype,
            noData=0,
            creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
        )

    # Use numpy.angle to extract the phase component of the complex wrapped interferogram
    wrapped_phase = ISCE2Dataset('filt_topophase.flat.geo', 'wrapped_phase', 1)
    cmd = (
        'gdal_calc.py '
        f'--outfile {product_name}/{product_name}_{wrapped_phase.suffix}.tif '
        f'-A merged/{wrapped_phase.name} --A_band={wrapped_phase.band} '
        '--calc angle(A) --type Float32 --format GTiff --NoDataValue=0 '
        '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
    )
    subprocess.run(cmd.split(' '), check=True)

    ds = gdal.Open('merged/los.rdr.geo', gdal.GA_Update)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.GetRasterBand(2).SetNoDataValue(0)
    del ds

    # Performs the inverse of the operation performed by MintPy:
    # https://github.com/insarlab/MintPy/blob/df96e0b73f13cc7e2b6bfa57d380963f140e3159/src/mintpy/objects/stackDict.py#L732-L737
    # First subtract the incidence angle from ninety degrees to go from sensor-to-ground to ground-to-sensor,
    # then convert to radians
    incidence_angle = ISCE2Dataset('los.rdr.geo', 'lv_theta', 1)
    cmd = (
        'gdal_calc.py '
        f'--outfile {product_name}/{product_name}_{incidence_angle.suffix}.tif '
        f'-A merged/{incidence_angle.name} --A_band={incidence_angle.band} '
        '--calc (90-A)*pi/180 --type Float32 --format GTiff --NoDataValue=0 '
        '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
    )
    subprocess.run(cmd.split(' '), check=True)

    # Performs the inverse of the operation performed by MintPy:
    # https://github.com/insarlab/MintPy/blob/df96e0b73f13cc7e2b6bfa57d380963f140e3159/src/mintpy/objects/stackDict.py#L739-L745
    # First add ninety degrees to the azimuth angle to go from angle-from-east to angle-from-north,
    # then convert to radians
    azimuth_angle = ISCE2Dataset('los.rdr.geo', 'lv_phi', 2)
    cmd = (
        'gdal_calc.py '
        f'--outfile {product_name}/{product_name}_{azimuth_angle.suffix}.tif '
        f'-A merged/{azimuth_angle.name} --A_band={azimuth_angle.band} '
        '--calc (90+A)*pi/180 --type Float32 --format GTiff --NoDataValue=0 '
        '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
    )
    subprocess.run(cmd.split(' '), check=True)

    ds = gdal.Open('merged/filt_topophase.unw.geo')
    geotransform = ds.GetGeoTransform()
    del ds

    epsg = utm_from_lon_lat(geotransform[0], geotransform[3])
    files = [str(path) for path in Path(product_name).glob('*.tif') if not path.name.endswith('rdr.tif')]
    for file in files:
        gdal.Warp(
            file,
            file,
            dstSRS=f'epsg:{epsg}',
            creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
            xRes=pixel_size,
            yRes=pixel_size,
            targetAlignedPixels=True,
        )


def make_parameter_file(
    out_path: Path,
    reference_scene: str,
    secondary_scene: str,
    azimuth_looks: int,
    range_looks: int,
    apply_water_mask: bool,
    dem_name: str = 'GLO_30',
    dem_resolution: int = 30,
) -> None:
    """Create a parameter file for the output product

    Args:
        out_path: path to output the parameter file
        reference_scene: Reference burst name
        secondary_scene: Secondary burst name
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks
        dem_name: Name of the DEM that is use
        dem_resolution: Resolution of the DEM

    returns:
        None
    """
    SPEED_OF_LIGHT = 299792458.0
    SPACECRAFT_HEIGHT = 693000.0
    EARTH_RADIUS = 6337286.638938101

    parser = etree.XMLParser(encoding='utf-8', recover=True)

    ref_tag = reference_scene[-4::]
    sec_tag = secondary_scene[-4::]
    print(ref_tag, sec_tag)
    reference_safe = [file for file in os.listdir('.') if file.endswith(f'{ref_tag}.SAFE')][0]
    secondary_safe = [file for file in os.listdir('.') if file.endswith(f'{sec_tag}.SAFE')][0]

    ref_annotation_path = f'{reference_safe}/annotation/'
    ref_annotation = [file for file in os.listdir(ref_annotation_path) if os.path.isfile(ref_annotation_path + file)][0]

    ref_manifest_xml = etree.parse(f'{reference_safe}/manifest.safe', parser)
    sec_manifest_xml = etree.parse(f'{secondary_safe}/manifest.safe', parser)
    ref_annotation_xml = etree.parse(f'{ref_annotation_path}{ref_annotation}', parser)
    topsProc_xml = etree.parse('topsProc.xml', parser)
    topsApp_xml = etree.parse('topsApp.xml', parser)

    safe = '{http://www.esa.int/safe/sentinel-1.0}'
    s1 = '{http://www.esa.int/safe/sentinel-1.0/sentinel-1}'
    metadata_path = './/metadataObject[@ID="measurementOrbitReference"]//xmlData//'
    orbit_number_query = metadata_path + safe + 'orbitNumber'
    orbit_direction_query = metadata_path + safe + 'extension//' + s1 + 'pass'

    ref_orbit_number = ref_manifest_xml.find(orbit_number_query).text
    ref_orbit_direction = ref_manifest_xml.find(orbit_direction_query).text
    sec_orbit_number = sec_manifest_xml.find(orbit_number_query).text
    sec_orbit_direction = sec_manifest_xml.find(orbit_direction_query).text
    ref_heading = float(ref_annotation_xml.find('.//platformHeading').text)
    ref_time = ref_annotation_xml.find('.//productFirstLineUtcTime').text
    slant_range_time = float(ref_annotation_xml.find('.//slantRangeTime').text)
    range_sampling_rate = float(ref_annotation_xml.find('.//rangeSamplingRate').text)
    number_samples = int(ref_annotation_xml.find('.//swathTiming/samplesPerBurst').text)
    baseline_perp = topsProc_xml.find('.//IW-2_Bperp_at_midrange_for_first_common_burst').text
    unwrapper_type = topsApp_xml.find('.//property[@name="unwrapper name"]').text
    phase_filter_strength = topsApp_xml.find('.//property[@name="filter strength"]').text

    slant_range_near = float(slant_range_time) * SPEED_OF_LIGHT / 2
    range_pixel_spacing = SPEED_OF_LIGHT / (2 * range_sampling_rate)
    slant_range_far = slant_range_near + (number_samples - 1) * range_pixel_spacing
    slant_range_center = (slant_range_near + slant_range_far) / 2

    s = ref_time.split('T')[1].split(':')
    utc_time = ((int(s[0]) * 60 + int(s[1])) * 60) + float(s[2])

    parameter_file = ParameterFile(
        reference_granule=reference_scene,
        secondary_granule=secondary_scene,
        reference_orbit_direction=ref_orbit_direction,
        reference_orbit_number=ref_orbit_number,
        secondary_orbit_direction=sec_orbit_direction,
        secondary_orbit_number=sec_orbit_number,
        baseline=float(baseline_perp),
        utc_time=utc_time,
        heading=ref_heading,
        spacecraft_height=SPACECRAFT_HEIGHT,
        earth_radius_at_nadir=EARTH_RADIUS,
        slant_range_near=slant_range_near,
        slant_range_center=slant_range_center,
        slant_range_far=slant_range_far,
        range_looks=int(range_looks),
        azimuth_looks=int(azimuth_looks),
        insar_phase_filter=True,
        phase_filter_parameter=float(phase_filter_strength),
        range_bandpass_filter=False,
        azimuth_bandpass_filter=False,
        dem_source=dem_name,
        dem_resolution=dem_resolution,
        unwrapping_type=unwrapper_type,
        speckle_filter=True,
        water_mask=apply_water_mask,
    )
    parameter_file.write(out_path)


def make_readme(
    product_dir: Path,
    product_name: str,
    reference_scene: str,
    secondary_scene: str,
    range_looks: int,
    azimuth_looks: int,
    apply_water_mask: bool,
) -> None:
    wrapped_phase_path = product_dir / f'{product_name}_wrapped_phase.tif'
    info = gdal.Info(str(wrapped_phase_path), format='json')
    secondary_granule_datetime_str = secondary_scene.split('_')[5]

    payload = {
        'processing_date': datetime.now(timezone.utc),
        'plugin_name': hyp3_isce2.__name__,
        'plugin_version': hyp3_isce2.__version__,
        'processor_name': isce.__name__.upper(),
        'processor_version': isce.__version__,
        'projection': get_projection(info['coordinateSystem']['wkt']),
        'pixel_spacing': info['geoTransform'][1],
        'product_name': product_name,
        'reference_burst_name': reference_scene,
        'secondary_burst_name': secondary_scene,
        'range_looks': range_looks,
        'azimuth_looks': azimuth_looks,
        'secondary_granule_date': datetime.strptime(secondary_granule_datetime_str, '%Y%m%dT%H%M%S'),
        'dem_name': 'GLO-30',
        'dem_pixel_spacing': '30 m',
        'apply_water_mask': apply_water_mask,
    }
    content = hyp3_isce2.metadata.util.render_template('insar_burst/insar_burst_readme.md.txt.j2', payload)

    output_file = product_dir / f'{product_name}_README.md.txt'
    with open(output_file, 'w') as f:
        f.write(content)


def main():
    """HyP3 entrypoint for the TOPS burst merging workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--reference', nargs='*', help='List of granules for the reference bursts')
    parser.add_argument('--secondary', nargs='*', help='List of granules for the secondary bursts')
    parser.add_argument(
        '--looks', choices=['20x4', '10x2', '5x1'], default='20x4', help='Number of looks to take in range and azimuth'
    )
    parser.add_argument(
        '--apply-water-mask',
        type=string_is_true,
        default=False,
        help='Apply a water body mask before unwrapping.',
    )
    args = parser.parse_args()
    granules_ref = list(set(args.reference))
    granules_sec = list(set(args.secondary))

    ids_ref = [
        granule.split('_')[1] + '_' + granule.split('_')[2] + '_' + granule.split('_')[4] for granule in granules_ref
    ]
    ids_sec = [
        granule.split('_')[1] + '_' + granule.split('_')[2] + '_' + granule.split('_')[4] for granule in granules_sec
    ]

    if len(list(set(ids_ref) - set(ids_sec))) > 0:
        raise Exception(
            'The reference bursts '
            + ', '.join(list(set(ids_ref) - set(ids_sec)))
            + ' do not have the correspondant bursts in the secondary granules'
        )
    elif len(list(set(ids_sec) - set(ids_ref))) > 0:
        raise Exception(
            'The secondary bursts '
            + ', '.join(list(set(ids_sec) - set(ids_ref)))
            + ' do not have the correspondant bursts in the reference granules'
        )

    if not granules_ref[0].split('_')[4] == granules_sec[0].split('_')[4]:
        raise Exception('The secondary and reference granules do not have the same polarization')

    if granules_ref[0].split('_')[3] > granules_sec[0].split('_')[3]:
        log.info('The secondary granules have a later date than the reference granules.')
        temp = copy.copy(granules_ref)
        granules_ref = copy.copy(granules_sec)
        granules_sec = temp

    swaths = list(set([int(granule.split('_')[2][2]) for granule in granules_ref]))

    reference_scene = burst2safe(granules_ref)
    reference_scene = os.path.basename(reference_scene).split('.')[0]

    secondary_scene = burst2safe(granules_sec)
    secondary_scene = os.path.basename(secondary_scene).split('.')[0]

    polarization = granules_ref[0].split('_')[4]

    range_looks, azimuth_looks = [int(looks) for looks in args.looks.split('x')]
    apply_water_mask = args.apply_water_mask

    insar_tops(reference_scene, secondary_scene, swaths, polarization, download=False)

    pixel_size = get_pixel_size(args.looks)
    product_name = get_product_name(reference_scene, secondary_scene, pixel_spacing=int(pixel_size))

    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    translate_outputs(product_name, pixel_size=pixel_size, include_radar=True)

    unwrapped_phase = f'{product_name}/{product_name}_unw_phase.tif'
    water_mask = f'{product_name}/{product_name}_water_mask.tif'

    if apply_water_mask:
        convert_raster_from_isce2_gdal('water_mask.wgs84', unwrapped_phase, water_mask)
        cmd = (
            'gdal_calc.py '
            f'--outfile {unwrapped_phase} '
            f'-A {unwrapped_phase} -B {water_mask} '
            '--calc A*B '
            '--overwrite '
            '--NoDataValue 0 '
            '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
        )
        subprocess.run(cmd.split(' '), check=True)

    make_browse_image(unwrapped_phase, f'{product_name}/{product_name}_unw_phase.png')

    make_readme(
        product_dir=product_dir,
        product_name=product_name,
        reference_scene=reference_scene,
        secondary_scene=secondary_scene,
        range_looks=range_looks,
        azimuth_looks=azimuth_looks,
        apply_water_mask=apply_water_mask,
    )
    make_parameter_file(
        Path(f'{product_name}/{product_name}.txt'),
        reference_scene=reference_scene,
        secondary_scene=secondary_scene,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        apply_water_mask=apply_water_mask,
    )
    output_zip = make_archive(base_name=product_name, format='zip', base_dir=product_name)

    if args.bucket:
        for browse in product_dir.glob('*.png'):
            create_thumbnail(browse, output_dir=product_dir)

        upload_file_to_s3(Path(output_zip), args.bucket, args.bucket_prefix)

        for product_file in product_dir.iterdir():
            upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)


if __name__ == '__main__':
    main()
