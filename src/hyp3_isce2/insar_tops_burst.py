"""Create a single-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import json
import logging
import os
import site
import subprocess
import sys
from pathlib import Path
from shutil import make_archive

import numpy as np
from hyp3lib.aws import upload_file_to_s3
from hyp3lib.get_orb import downloadSentinelOrbitFile
from osgeo import gdal
from osgeo import osr

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


# TODO is this the format we want?
# TODO unit test
def get_product_name(
        reference_scene: str,
        secondary_scene: str,
        reference_burst_number: int,
        secondary_burst_number: int,
        swath_number: int,
        polarization: str) -> str:
    reference_name = f'{reference_scene}_IW{swath_number}_{polarization}_{reference_burst_number}'
    secondary_name = f'{secondary_scene}_IW{swath_number}_{polarization}_{secondary_burst_number}'
    return f'{reference_name}x{secondary_name}'


# TODO add more parameters
# TODO does the format need to be the same as for our INSAR_GAMMA products?
# TODO unit test
def make_parameter_file(
        out_path: Path,
        reference_scene: str,
        secondary_scene: str) -> None:
    output = {
        'reference_scene': reference_scene,
        'secondary_scene': secondary_scene,
    }
    with out_path.open('w') as f:
        json.dump(output, f)


def make_tiff(infile, outfile, reffile, bn=1, nodata=-1):
    ds = gdal.Open(infile)

    band = ds.GetRasterBand(bn)

    data = band.ReadAsArray()

    [cols, rows] = data.shape

    projection = osr.SpatialReference()

    projection.ImportFromWkt(ds.GetProjectionRef())

    # read ref_file

    ds_ref = gdal.Open(reffile, gdal.GA_ReadOnly)

    data_ref = ds_ref.GetRasterBand(1).ReadAsArray()

    # change the datatype and apply the mask with nodata

    data = data.astype(dtype=np.float32)

    data[data_ref == 0] = nodata

    driver = gdal.GetDriverByName("GTiff")

    des = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)

    des.SetGeoTransform(ds.GetGeoTransform())

    des.SetProjection(projection.ExportToWkt())

    outband = des.GetRasterBand(1)

    outband.WriteArray(data)

    des.GetRasterBand(1).SetNoDataValue(nodata)

    des = None


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
        args.polarization
    )
    os.mkdir(product_name)

    # TODO need to set nodata values
    # TODO what output projection do we want? currently EPSG:4326
    gdal.Translate(
        destName=f'{product_name}/{product_name}_unw_phase.tif',
        srcDS=str(product_dir / 'filt_topophase.unw.geo'),
        bandList=[2],
        noData=0,
        creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
    )
    gdal.Translate(
        destName=f'{product_name}/{product_name}_corr.tif',
        srcDS=str(product_dir / 'phsig.cor.geo'),
        noData=0,
        creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
    )

    make_tiff(str(product_dir / 'filt_topophase.unw.conncomp.geo'),
              f'{product_name}/{product_name}_conn_comp.tif',
              str(product_dir / 'filt_topophase.unw.geo')
              )

    subprocess.call([
        'gdal_calc.py',
        '--outfile', f'{product_name}/{product_name}_wrapped_phase.tif',
        '-A', str(product_dir / 'filt_topophase.flat.geo'),
        '--calc', 'angle(A)',
        '--NoDataValue=0',
        '--type', 'Float32',
        '--creation-option', 'TILED=YES',
        '--creation-option', 'COMPRESS=LZW',
        '--creation-option', 'NUM_THREADS=ALL_CPUS',
    ])

    make_parameter_file(
        Path(f'{product_name}/{product_name}.json'),
        args.reference_scene,
        args.secondary_scene,
    )
    product_file = make_archive(base_name=product_name, format='zip', base_dir=product_name)

    if args.bucket:
        upload_file_to_s3(Path(product_file), args.bucket, args.bucket_prefix)
