"""Create a single-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import logging
import os
import site
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from shutil import copyfile, make_archive

import numpy as np
from hyp3lib.aws import upload_file_to_s3
from hyp3lib.get_orb import downloadSentinelOrbitFile
from hyp3lib.image import create_thumbnail
from lxml import etree
from osgeo import gdal

from hyp3_isce2 import topsapp
from hyp3_isce2.burst import (
    BurstParams,
    download_bursts,
    get_isce2_burst_bbox,
    get_product_name,
    get_region_of_interest,
)
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.logging import configure_root_logger
from hyp3_isce2.s1_auxcal import download_aux_cal
from hyp3_isce2.utils import make_browse_image, utm_from_lon_lat

gdal.UseExceptions()

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
    ref_footprint = get_isce2_burst_bbox(ref_params)
    sec_footprint = get_isce2_burst_bbox(sec_params)

    insar_roi = get_region_of_interest(ref_footprint, sec_footprint, is_ascending=is_ascending)
    dem_roi = ref_footprint.intersection(sec_footprint).bounds
    log.info(f'InSAR ROI: {insar_roi}')
    log.info(f'DEM ROI: {dem_roi}')

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
        swaths=swath_number,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )
    config_path = config.write_template('topsApp.xml')

    topsapp.run_topsapp_burst(start='startup', end='preprocess', config_xml=config_path)
    topsapp.swap_burst_vrts()
    topsapp.run_topsapp_burst(start='computeBaselines', end='unwrap2stage', config_xml=config_path)
    copyfile('merged/z.rdr.full.xml', 'merged/z.rdr.full.vrt.xml')
    topsapp.run_topsapp_burst(start='geocode', end='geocode', config_xml=config_path)

    return Path('merged')


def make_parameter_file(
    out_path: Path,
    reference_scene: str,
    secondary_scene: str,
    swath_number: int,
    reference_burst_number: int,
    secondary_burst_number: int,
    polarization: str = 'VV',
    azimuth_looks: int = 4,
    range_looks: int = 20,
    dem_name: str = 'GLO_30',
    dem_resolution: int = 30
) -> None:
    """Create a parameter file for the output product

    Args:
        out_path: path to output the parameter file
        reference_scene: Reference SLC name
        secondary_scene: Secondary SLC name
        swath_number: Number of swath to grab bursts from (1, 2, or 3) for IW
        reference_burst_number: Number of burst to download for reference (0-indexed from first collect)
        secondary_burst_number: Number of burst to download for secondary (0-indexed from first collect)
        polarization: Polarization to use
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

    ref_annotation_path = f'{reference_scene}.SAFE/annotation/'
    ref_annotation = [file for file in os.listdir(ref_annotation_path) if os.path.isfile(ref_annotation_path + file)][0]

    ref_manifest_xml = etree.parse(f'{reference_scene}.SAFE/manifest.safe', parser)
    sec_manifest_xml = etree.parse(f'{secondary_scene}.SAFE/manifest.safe', parser)
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
    # FIXME: Not sure why baseline is negative in some cases. I believe it is just a convention
    baseline_perp = np.abs(topsProc_xml.find(f'.//IW-{swath_number}_Bperp_at_midrange_for_first_common_burst').text)
    unwrapper_type = topsApp_xml.find('.//property[@name="unwrapper name"]').text
    phase_filter_strength = topsApp_xml.find('.//property[@name="filter strength"]').text

    slant_range_near = float(slant_range_time) * SPEED_OF_LIGHT / 2
    range_pixel_spacing = SPEED_OF_LIGHT / (2 * range_sampling_rate)
    slant_range_far = slant_range_near + (number_samples - 1) * range_pixel_spacing
    slant_range_center = (slant_range_near + slant_range_far) / 2

    s = ref_time.split('T')[1].split(':')
    utc_time = (int(s[0]) * 60 + int(s[1]) * 60) + float(s[2])

    output_strings = [
        f'Reference Granule: {reference_scene}\n',
        f'Secondary Granule: {secondary_scene}\n',
        f'Reference Pass Direction: {ref_orbit_direction}\n',
        f'Reference Orbit Number: {ref_orbit_number}\n',
        f'Secondary Pass Direction: {sec_orbit_direction}\n',
        f'Secondary Orbit Number: {sec_orbit_number}\n',
        f'Reference Burst Number: {reference_burst_number}\n',
        f'Secondary Burst Number: {secondary_burst_number}\n',
        f'Swath Number: {swath_number}\n',
        f'Polarization: {polarization}\n',
        f'Baseline: {baseline_perp}\n',
        f'UTC time: {utc_time}\n',
        f'Heading: {ref_heading}\n',
        f'Spacecraft height: {SPACECRAFT_HEIGHT}\n',
        f'Earth radius at nadir: {EARTH_RADIUS}\n',
        f'Slant range near: {slant_range_near}\n',
        f'Slant range center: {slant_range_center}\n',
        f'Slant range far: {slant_range_far}\n',
        f'Range looks: {range_looks}\n',
        f'Azimuth looks: {azimuth_looks}\n',
        'INSAR phase filter: yes\n',
        f'Phase filter parameter: {phase_filter_strength}\n',
        'Range bandpass filter: no\n',
        'Azimuth bandpass filter: no\n',
        f'DEM source: {dem_name}\n',
        f'DEM resolution (m): {dem_resolution}\n',
        f'Unwrapping type: {unwrapper_type}\n',
        'Speckle filter: yes\n'
    ]

    output_string = "".join(output_strings)

    with open(out_path.__str__(), 'w') as outfile:
        outfile.write(output_string)


def translate_outputs(isce_output_dir: Path, product_name: str):
    """Translate ISCE outputs to a standard GTiff format with a UTM projection

    Args:
        isce_output_dir: Path to the ISCE output directory
        product_name: Name of the product
    """
    ISCE2Dataset = namedtuple('ISCE2Dataset', ['name', 'suffix', 'band'])
    datasets = [
        ISCE2Dataset('filt_topophase.unw.geo', 'unw_phase', 2),
        ISCE2Dataset('phsig.cor.geo', 'corr', 1),
        ISCE2Dataset('z.rdr.full.geo', 'dem', 1),
        ISCE2Dataset('filt_topophase.unw.conncomp.geo', 'conncomp', 1),
    ]

    for dataset in datasets:
        out_file = str(Path(product_name) / f'{product_name}_{dataset.suffix}.tif')
        in_file = str(isce_output_dir / dataset.name)

        gdal.Translate(
            destName=out_file,
            srcDS=in_file,
            bandList=[dataset.band],
            format='GTiff',
            noData=0,
            creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
        )

    # Use numpy.angle to extract the phase component of the complex wrapped interferogram
    wrapped_phase = ISCE2Dataset('filt_topophase.flat.geo', 'wrapped_phase', 1)
    cmd = (
        'gdal_calc.py '
        f'--outfile {product_name}/{product_name}_{wrapped_phase.suffix}.tif '
        f'-A {isce_output_dir / wrapped_phase.name} --A_band={wrapped_phase.band} '
        '--calc angle(A) --type Float32 --format GTiff --NoDataValue=0 '
        '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
    )
    subprocess.check_call(cmd.split(' '))

    ds = gdal.Open(str(isce_output_dir / 'los.rdr.geo'), gdal.GA_Update)
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
        f'-A {isce_output_dir / incidence_angle.name} --A_band={incidence_angle.band} '
        '--calc (90-A)*pi/180 --type Float32 --format GTiff --NoDataValue=0 '
        '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
    )
    subprocess.check_call(cmd.split(' '))

    # Performs the inverse of the operation performed by MintPy:
    # https://github.com/insarlab/MintPy/blob/df96e0b73f13cc7e2b6bfa57d380963f140e3159/src/mintpy/objects/stackDict.py#L739-L745
    # First add ninety degrees to the azimuth angle to go from angle-from-east to angle-from-north,
    # then convert to radians
    azimuth_angle = ISCE2Dataset('los.rdr.geo', 'lv_phi', 2)
    cmd = (
        'gdal_calc.py '
        f'--outfile {product_name}/{product_name}_{azimuth_angle.suffix}.tif '
        f'-A {isce_output_dir / azimuth_angle.name} --A_band={azimuth_angle.band} '
        '--calc (90+A)*pi/180 --type Float32 --format GTiff --NoDataValue=0 '
        '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
    )
    subprocess.check_call(cmd.split(' '))

    ds = gdal.Open(str(isce_output_dir / 'filt_topophase.unw.geo'))
    geotransform = ds.GetGeoTransform()
    del ds

    epsg = utm_from_lon_lat(geotransform[0], geotransform[3])
    files = [str(path) for path in Path(product_name).glob('*.tif')]
    for file in files:
        gdal.Warp(
            file,
            file,
            dstSRS=f'epsg:{epsg}',
            creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
        )

    make_browse_image(f'{product_name}/{product_name}_unw_phase.tif', f'{product_name}/{product_name}_unw_phase.png')


def main():
    """HyP3 entrypoint for the burst TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument('--reference-scene', type=str, required=True)
    parser.add_argument('--secondary-scene', type=str, required=True)
    parser.add_argument('--swath-number', type=int, required=True)
    parser.add_argument('--polarization', type=str, default='VV')
    parser.add_argument('--reference-burst-number', type=int, required=True)
    parser.add_argument('--secondary-burst-number', type=int, required=True)
    parser.add_argument('--azimuth-looks', type=int, default=4)
    parser.add_argument('--range-looks', type=int, default=20)

    args = parser.parse_args()

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    log.info('Begin ISCE2 TopsApp run')

    isce_output_dir = insar_tops_burst(
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

    product_name = get_product_name(
        args.reference_scene,
        args.secondary_scene,
        args.reference_burst_number,
        args.secondary_burst_number,
        args.swath_number,
        args.polarization,
    )

    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    translate_outputs(isce_output_dir, product_name)
    make_parameter_file(
        Path(f'{product_name}/{product_name}.txt'),
        reference_scene=args.reference_scene,
        secondary_scene=args.secondary_scene,
        swath_number=args.swath_number,
        polarization=args.polarization,
        reference_burst_number=args.reference_burst_number,
        secondary_burst_number=args.secondary_burst_number,
        azimuth_looks=args.azimuth_looks,
        range_looks=args.range_looks
    )
    output_zip = make_archive(base_name=product_name, format='zip', base_dir=product_name)

    if args.bucket:
        for browse in product_dir.glob('*.png'):
            create_thumbnail(browse, output_dir=product_dir)

        upload_file_to_s3(Path(output_zip), args.bucket, args.bucket_prefix)

        for product_file in product_dir.iterdir():
            upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
