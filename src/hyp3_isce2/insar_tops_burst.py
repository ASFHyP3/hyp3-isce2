"""Create a single-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile, make_archive
from typing import Iterable, Optional

import isce
from hyp3lib.aws import upload_file_to_s3
from hyp3lib.image import create_thumbnail
from hyp3lib.util import string_is_true
from isceobj.TopsProc.runMergeBursts import multilook
from lxml import etree
from osgeo import gdal, gdalconst
from pyproj import CRS
from s1_orbits import fetch_for_scene

import hyp3_isce2
import hyp3_isce2.metadata.util
from hyp3_isce2 import topsapp
from hyp3_isce2.burst import (
    BurstPosition,
    download_bursts,
    get_burst_params,
    get_isce2_burst_bbox,
    get_product_name,
    get_region_of_interest,
    multilook_radar_merge_inputs,
    validate_bursts,
)
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.logger import configure_root_logger
from hyp3_isce2.s1_auxcal import download_aux_cal
from hyp3_isce2.utils import (
    ParameterFile,
    image_math,
    isce2_copy,
    make_browse_image,
    oldest_granule_first,
    resample_to_radar_io,
    utm_from_lon_lat,
)
from hyp3_isce2.water_mask import create_water_mask


gdal.UseExceptions()

log = logging.getLogger(__name__)


@dataclass
class ISCE2Dataset:
    name: str
    suffix: str
    band: Iterable[int]
    dtype: Optional[int] = gdalconst.GDT_Float32


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
    secondary_granule_datetime_str = secondary_scene.split('_')[3]

    payload = {
        'processing_date': datetime.now(timezone.utc),
        'plugin_name': hyp3_isce2.__name__,
        'plugin_version': hyp3_isce2.__version__,
        'processor_name': isce.__name__.upper(),
        'processor_version': isce.__version__,
        'projection': hyp3_isce2.metadata.util.get_projection(info['coordinateSystem']['wkt']),
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


def make_parameter_file(
    out_path: Path,
    reference_scene: str,
    secondary_scene: str,
    swath_number: int,
    azimuth_looks: int,
    range_looks: int,
    multilook_position: BurstPosition,
    apply_water_mask: bool,
    dem_name: str = 'GLO_30',
    dem_resolution: int = 30,
) -> None:
    """Create a parameter file for the output product

    Args:
        out_path: path to output the parameter file
        reference_scene: Reference burst name
        secondary_scene: Secondary burst name
        swath_number: Number of swath to grab bursts from (1, 2, or 3) for IW
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

    ref_tag = reference_scene[-10:-6]
    sec_tag = secondary_scene[-10:-6]
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
    baseline_perp = topsProc_xml.find(f'.//IW-{swath_number}_Bperp_at_midrange_for_first_common_burst').text
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
        radar_n_lines=multilook_position.n_lines,
        radar_n_samples=multilook_position.n_samples,
        radar_first_valid_line=multilook_position.first_valid_line,
        radar_n_valid_lines=multilook_position.n_valid_lines,
        radar_first_valid_sample=multilook_position.first_valid_sample,
        radar_n_valid_samples=multilook_position.n_valid_samples,
        multilook_azimuth_time_interval=multilook_position.azimuth_time_interval,
        multilook_range_pixel_size=multilook_position.range_pixel_size,
        radar_sensing_stop=multilook_position.sensing_stop,
    )
    parameter_file.write(out_path)


def find_product(pattern: str) -> str:
    """Find a single file within the working directory's structure

    Args:
        pattern: Glob pattern for file

    Returns
        Path to file
    """
    search = Path.cwd().glob(pattern)
    product = str(list(search)[0])
    return product


def translate_outputs(product_name: str, pixel_size: float, include_radar: bool = False, use_multilooked=False) -> None:
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

    suffix = '01'
    if use_multilooked:
        suffix += '.multilooked'

    rdr_datasets = [
        ISCE2Dataset(
            find_product(f'fine_interferogram/IW*/burst_{suffix}.int.vrt'),
            'wrapped_phase_rdr',
            [1],
            gdalconst.GDT_CFloat32,
        ),
        ISCE2Dataset(find_product(f'geom_reference/IW*/lat_{suffix}.rdr.vrt'), 'lat_rdr', [1]),
        ISCE2Dataset(find_product(f'geom_reference/IW*/lon_{suffix}.rdr.vrt'), 'lon_rdr', [1]),
        ISCE2Dataset(find_product(f'geom_reference/IW*/los_{suffix}.rdr.vrt'), 'los_rdr', [1, 2]),
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


def get_pixel_size(looks: str) -> float:
    return {'20x4': 80.0, '10x2': 40.0, '5x1': 20.0}[looks]


def convert_raster_from_isce2_gdal(input_image, ref_image, output_image):
    """Convert the water mask in WGS84 to be the same projection and extent of the output product.

    Args:
        input_image: dem file name
        ref_image: output geotiff file name
        output_image: water mask file name
    """

    ref_ds = gdal.Open(ref_image)

    gt = ref_ds.GetGeoTransform()

    pixel_size = gt[1]

    minx = gt[0]
    maxx = gt[0] + gt[1] * ref_ds.RasterXSize
    maxy = gt[3]
    miny = gt[3] + gt[5] * ref_ds.RasterYSize

    crs = ref_ds.GetSpatialRef()
    epsg = CRS.from_wkt(crs.ExportToWkt()).to_epsg()

    del ref_ds

    gdal.Warp(
        output_image,
        input_image,
        dstSRS=f'epsg:{epsg}',
        creationOptions=['TILED=YES', 'COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],
        outputBounds=[minx, miny, maxx, maxy],
        xRes=pixel_size,
        yRes=pixel_size,
        targetAlignedPixels=True,
    )


def main():
    """HyP3 entrypoint for the burst TOPS workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bucket', help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', default='', help='Add a bucket prefix to product(s)')
    parser.add_argument(
        '--looks', choices=['20x4', '10x2', '5x1'], default='20x4', help='Number of looks to take in range and azimuth'
    )
    parser.add_argument(
        '--apply-water-mask',
        type=string_is_true,
        default=False,
        help='Apply a water body mask before unwrapping.',
    )
    # Allows granules to be given as a space-delimited list of strings (e.g. foo bar) or as a single
    # quoted string that contains spaces (e.g. "foo bar"). AWS Batch uses the latter format when
    # invoking the container command.
    parser.add_argument('granules', type=str.split, nargs='+')

    args = parser.parse_args()

    args.granules = [item for sublist in args.granules for item in sublist]
    if len(args.granules) != 2:
        parser.error('Must provide exactly two granules')

    configure_root_logger()
    log.debug(' '.join(sys.argv))

    log.info('Begin ISCE2 TopsApp run')

    reference_scene, secondary_scene = oldest_granule_first(args.granules[0], args.granules[1])
    validate_bursts(reference_scene, secondary_scene)
    swath_number = int(reference_scene[12])
    range_looks, azimuth_looks = [int(looks) for looks in args.looks.split('x')]
    apply_water_mask = args.apply_water_mask

    insar_tops_burst(
        reference_scene=reference_scene,
        secondary_scene=secondary_scene,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
        swath_number=swath_number,
        apply_water_mask=apply_water_mask,
    )

    log.info('ISCE2 TopsApp run completed successfully')

    multilook_position = multilook_radar_merge_inputs(swath_number, rg_looks=range_looks, az_looks=azimuth_looks)

    pixel_size = get_pixel_size(args.looks)
    product_name = get_product_name(reference_scene, secondary_scene, pixel_spacing=int(pixel_size))

    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    translate_outputs(product_name, pixel_size=pixel_size, include_radar=True, use_multilooked=True)

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
        swath_number=swath_number,
        multilook_position=multilook_position,
        apply_water_mask=apply_water_mask,
    )
    output_zip = make_archive(base_name=product_name, format='zip', base_dir=product_name)

    if args.bucket:
        for browse in product_dir.glob('*.png'):
            create_thumbnail(browse, output_dir=product_dir)

        upload_file_to_s3(Path(output_zip), args.bucket, args.bucket_prefix)

        for product_file in product_dir.iterdir():
            upload_file_to_s3(product_file, args.bucket, args.bucket_prefix)
