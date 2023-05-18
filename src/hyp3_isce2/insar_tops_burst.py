"""Create a single-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import json
import logging
import os
import site
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from shutil import copyfile, make_archive

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.get_orb import downloadSentinelOrbitFile
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
from hyp3_isce2.s1_auxcal import download_aux_cal
from hyp3_isce2.utils import make_browse_image, utm_from_lon_lat

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


# TODO add more parameters
# TODO does the format need to be the same as for our INSAR_GAMMA products?
# TODO unit test
def make_parameter_file(out_path: Path, reference_scene: str, secondary_scene: str) -> None:
    output = {
        'reference_scene': reference_scene,
        'secondary_scene': secondary_scene,
    }
    with out_path.open('w') as f:
        json.dump(output, f)


def translate_outputs(product_dir: Path, product_name: str):
    """Translate ISCE outputs to a standard GTiff format with a UTM projection

    Args:
        product_dir: Path to the ISCE merge directory
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
        in_file = str(product_dir / dataset.name)

        gdal.Translate(
            destName=out_file,
            srcDS=in_file,
            bandList=[dataset.band],
            format='GTiff',
            noData=0,
            creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
        )

    wrapped_phase = ISCE2Dataset('filt_topophase.flat.geo', 'wrapped_phase', 1)
    cmd = (
        'gdal_calc.py '
        f'--outfile {product_name}/{product_name}_{wrapped_phase.suffix}.tif '
        f'-A {product_dir / wrapped_phase.name} '
        '--calc angle(A) --type Float32 --format GTiff --NoDataValue=0 '
        '--creation-option TILED=YES --creation-option COMPRESS=LZW --creation-option NUM_THREADS=ALL_CPUS'
    )
    subprocess.check_call(cmd.split(' '))

    ds = gdal.Open(str(product_dir / 'filt_topophase.unw.geo'))
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

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
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

    product_name = get_product_name(
        args.reference_scene,
        args.secondary_scene,
        args.reference_burst_number,
        args.secondary_burst_number,
        args.swath_number,
        args.polarization,
    )

    Path(product_name).mkdir(parents=True, exist_ok=True)
    translate_outputs(product_dir, product_name)
    make_parameter_file(
        Path(f'{product_name}/{product_name}.json'),
        args.reference_scene,
        args.secondary_scene,
    )
    product_file = make_archive(base_name=product_name, format='zip', base_dir=product_name)

    if args.bucket:
        upload_file_to_s3(Path(product_file), args.bucket, args.bucket_prefix)
