"""Create a single-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import logging
import os
import site
import sys
from lxml import etree
from pathlib import Path
from shutil import make_archive

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.get_orb import downloadSentinelOrbitFile
from hyp3lib.image import create_thumbnail

from hyp3_isce2 import topsapp
from hyp3_isce2.burst import BurstParams, download_bursts, get_isce2_burst_bbox, get_region_of_interest
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
    ref_footprint = get_isce2_burst_bbox(ref_params)
    sec_footprint = get_isce2_burst_bbox(sec_params)

    insar_roi = get_region_of_interest(ref_footprint, sec_footprint, is_ascending=is_ascending)
    dem_roi = ref_footprint.intersection(sec_footprint).bounds
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


def write_parameters_file(
    reference_scene: str,
    secondary_scene: str,
    swath_number: int,
    reference_burst_number: int,
    secondary_burst_number: int,
    polarization: str = "VV",
    azimuth_looks: int = 4,
    range_looks: int = 20,
    dem_name: str = "GLO_30",
    dem_resolution: int = 30
):

    filepath = Path("parameters.txt")

    parser = etree.XMLParser(encoding='utf-8', recover=True)

    ref_xml_tree  = etree.parse(f'{reference_scene}.SAFE/manifest.safe', parser)
    sec_xml_tree  = etree.parse(f'{secondary_scene}.SAFE/manifest.safe', parser)
    proc_xml_tree = etree.parse(f'topsProc.xml', parser)
    app_xml_tree  = etree.parse(f'topsApp.xml', parser)

    safe = '{http://www.esa.int/safe/sentinel-1.0}'
    s1   = '{http://www.esa.int/safe/sentinel-1.0/sentinel-1}'
    metadata_path = './/metadataObject[@ID="measurementOrbitReference"]//xmlData//'
    orbit_number_query = metadata_path + safe + 'orbitNumber'
    orbit_direction_query = metadata_path + safe + 'extension//' + s1 + 'pass'

    ref_orbit_number    = ref_xml_tree.find(orbit_number_query).text
    ref_orbit_direction = ref_xml_tree.find(orbit_direction_query).text
    sec_orbit_number    = sec_xml_tree.find(orbit_number_query).text
    sec_orbit_direction = sec_xml_tree.find(orbit_direction_query).text
    baseline_par  = proc_xml_tree.find('.//IW-2_Bpar_at_midrange_for_first_common_burst').text
    baseline_perp = proc_xml_tree.find('.//IW-2_Bperp_at_midrange_for_first_common_burst').text
    unwrapper_type = app_xml_tree.find('.//property[@name="unwrapper name"]').text
    phase_filter_strength = app_xml_tree.find('.//property[@name="filter strength"]').text

    output_strings = [
        f'Reference Scene: {reference_scene}\n',
        f'Secondary Scene: {secondary_scene}\n',
        f'Reference Pass Direction: {ref_orbit_direction}\n',
        f'Reference Orbit Number: {ref_orbit_number}\n',
        f'Secondary Pass Direction: {sec_orbit_direction}\n',
        f'Secondary Orbit Number: {sec_orbit_number}\n',
        f'Reference Burst Number: {reference_burst_number}\n',
        f'Secondary Burst Number: {secondary_burst_number}\n',
        f'Swath Number: {swath_number}\n',
        f'Polarization: {polarization}\n',
        f'Parallel Baseline: {baseline_par}\n',
        f'Perpindicular Baseline: {baseline_perp}\n',
        f'UTC time: \n',
        f'Heading: \n',
        f'Spacecraft height: 693000.0\n',
        f'Earth radius at nadir: 6337286.638938101\n',
        f'Slant range near: \n',
        f'Slant range center: \n',
        f'Slant range far: \n',
        f'Range looks: {range_looks}\n',
        f'Azimuth looks: {azimuth_looks}\n',
        f'INSAR phase filter: yes\n',
        f'Phase filter parameter: {phase_filter_strength}\n',
        f'Resolution of output (m): \n',
        f'Range bandpass filter: no\n',
        f'Azimuth bandpass filter: no\n',
        f'DEM source: {dem_name}\n',
        f'DEM resolution (m): {dem_resolution}\n',
        f'Unwrapping type: {unwrapper_type}\n',
        f'Phase at reference point: \n',
        f'Azimuth line of the reference point in SAR space: \n',
        f'Range pixel of the reference point in SAR space: \n',
        f'Y coordinate of the reference point in the map projection: \n',
        f'X coordinate of the reference point in the map projection: \n',
        f'Latitude of the reference point (WGS84): \n',
        f'Longitude of the reference point (WGS84): \n',
        f'Unwrapping threshold: \n',
        f'Speckle filter: yes\n'
    ]

    output_string = "".join(output_strings)

    with open(filepath.__str__(), 'w') as outfile:
        outfile.write(output_string)

    return filepath


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

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    log.debug(' '.join(sys.argv))

    # product_dir = insar_tops_burst(
    #     reference_scene=args.reference_scene,
    #     secondary_scene=args.secondary_scene,
    #     swath_number=args.swath_number,
    #     polarization=args.polarization,
    #     reference_burst_number=args.reference_burst_number,
    #     secondary_burst_number=args.secondary_burst_number,
    #     azimuth_looks=args.azimuth_looks,
    #     range_looks=args.range_looks,
    # )

    log.info('ISCE2 TopsApp run completed successfully')

    write_parameters_file(
        reference_scene=args.reference_scene,
        secondary_scene=args.secondary_scene,
        swath_number=args.swath_number,
        polarization=args.polarization,
        reference_burst_number=args.reference_burst_number,
        secondary_burst_number=args.secondary_burst_number,
        azimuth_looks=args.azimuth_looks,
        range_looks=args.range_looks
    )

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
