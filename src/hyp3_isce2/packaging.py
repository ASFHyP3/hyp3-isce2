import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex

import isce
from hyp3lib.aws import upload_file_to_s3
from hyp3lib.image import create_thumbnail
from lxml import etree
from osgeo import gdal, gdalconst
from pyproj import CRS

import hyp3_isce2
import hyp3_isce2.metadata.util
from hyp3_isce2.utils import ParameterFile, get_projection, utm_from_lon_lat


@dataclass
class ISCE2Dataset:
    name: str
    suffix: str
    band: int | list[int]
    dtype: int = gdalconst.GDT_Float32


def get_relative_orbit(reference_safe_path: Path) -> int:
    parser = etree.XMLParser(encoding='utf-8', recover=True)
    manifest_xml = etree.parse(reference_safe_path / 'manifest.safe', parser)
    orbit_number = manifest_xml.find(
        './/metadataObject[@ID="measurementOrbitReference"]//xmlData//'
        '{http://www.esa.int/safe/sentinel-1.0}relativeOrbitNumber'
    ).text  # type: ignore[union-attr]
    assert orbit_number is not None
    return int(orbit_number)


def get_pixel_size(range_looks: int, azimuth_looks: int) -> int:
    return max(range_looks, azimuth_looks * 5) * 4


def find_product(pattern: str) -> str:
    """Find a single file within the working directory's structure

    Args:
        pattern: Glob pattern for file

    Returns:
        Path to file
    """
    search = Path.cwd().glob(pattern)
    product = str(list(search)[0])
    return product


def _get_subswath_string(reference_scenes: list[str], swath_number: str) -> str:
    scenes = [scene for scene in reference_scenes if scene.split('_')[2][2] == swath_number]
    first_burst_number = min(scenes).split('_')[1] if scenes else '000000'
    scene_count = len(scenes)
    return f'{first_burst_number}s{swath_number}n{scene_count:02d}'


def _get_burst_date(scene: str) -> str:
    return scene.split('_')[3].split('T')[0]


def get_product_name(
    reference_scenes: list[str],
    secondary_scenes: list[str],
    relative_orbit: int,
    pixel_spacing: int,
    polarization: str,
) -> str:
    """Get the name of the interferogram product.

    Args:
        reference_scenes: List of the reference burst granule names.
        secondary_scenes: List of the secondary burst granule names.
        relative_orbit: Relative orbit number of the input scenes.
        pixel_spacing: Spacing of the pixels in the output image.
        polarization: Polarization of the input scenes.

    Returns:
        The name of the interferogram product.
    """
    s1 = _get_subswath_string(reference_scenes, '1')
    s2 = _get_subswath_string(reference_scenes, '2')
    s3 = _get_subswath_string(reference_scenes, '3')

    reference_date = min(_get_burst_date(scene) for scene in reference_scenes)
    secondary_date = max(_get_burst_date(scene) for scene in secondary_scenes)

    product_id = token_hex(2).upper()
    return (
        f'S1_{relative_orbit:03d}_{s1}-{s2}-{s3}_IW_{reference_date}_{secondary_date}'
        f'_{polarization}_INT{pixel_spacing}_{product_id}'
    )


def translate_outputs(
    product_name: str,
    pixel_size: int,
) -> None:
    """Translate ISCE outputs to a standard GTiff format with a UTM projection.
    Assume you are in the top level of an ISCE run directory

    Args:
        product_name: Name of the product
        pixel_size: Pixel size
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
    files = [str(path) for path in Path(product_name).glob('*.tif')]
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


def water_mask(unwrapped_phase: str, water_mask: str) -> None:
    """Apply the water mask to the unwrapped phase

    Args:
        unwrapped_phase: The unwrapped phase file
        water_mask: The water mask file
    """
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


def _get_data_year(secondary_scenes: list[str]) -> str:
    max_date = max(_get_burst_date(scene) for scene in secondary_scenes)
    return max_date[:4]


def make_readme(
    product_dir: Path,
    product_name: str,
    reference_scenes: list[str],
    secondary_scenes: list[str],
    range_looks: int,
    azimuth_looks: int,
    apply_water_mask: bool,
) -> None:
    wrapped_phase_path = product_dir / f'{product_name}_wrapped_phase.tif'
    info = gdal.Info(str(wrapped_phase_path), format='json')

    data_year = _get_data_year(secondary_scenes)

    payload = {
        'processing_date': datetime.now(timezone.utc),
        'plugin_name': hyp3_isce2.__name__,
        'plugin_version': hyp3_isce2.__version__,
        'processor_name': isce.__name__.upper(),
        'processor_version': isce.__version__,
        'projection': get_projection(info['coordinateSystem']['wkt']),
        'pixel_spacing': info['geoTransform'][1],
        'product_name': product_name,
        'reference_scenes': reference_scenes,
        'secondary_scenes': secondary_scenes,
        'range_looks': range_looks,
        'azimuth_looks': azimuth_looks,
        'data_year': data_year,
        'dem_name': 'GLO-30',
        'dem_pixel_spacing': '30 m',
        'apply_water_mask': apply_water_mask,
    }
    content = hyp3_isce2.metadata.util.render_template('insar_burst/insar_burst_readme.md.txt.j2', payload)

    output_file = product_dir / f'{product_name}_README.md.txt'
    with open(output_file, 'w') as f:
        f.write(content)


def get_baseline_perp(topsProc_xml: etree._ElementTree) -> float:
    for swath in [1, 2, 3]:
        bperp_element = topsProc_xml.find(f'.//IW-{swath}_Bperp_at_midrange_for_first_common_burst')
        if bperp_element is not None:
            bperp_txt = bperp_element.text
            assert bperp_txt is not None
            return float(bperp_txt)

    raise ValueError('No Bperp found in topsProc.xml')


def make_parameter_file(
    out_path: Path,
    reference_scenes: list[str],
    secondary_scenes: list[str],
    reference_safe_path: Path,
    secondary_safe_path: Path,
    processing_path: Path,
    azimuth_looks: int,
    range_looks: int,
    apply_water_mask: bool,
    dem_name: str = 'GLO_30',
    dem_resolution: int = 30,
) -> None:
    """Create a parameter file for the output product

    Args:
        out_path: path to output the parameter file
        reference_scenes: List of reference scene names (full SLC or burst names)
        secondary_scenes: List of secondary scene names (full SLC or burst names)
        reference_safe_path: Path to the reference SAFE directory
        secondary_safe_path: Path to the secondary SAFE directory
        processing_path: Path to the processing directory
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks
        dem_name: Name of the DEM that is use
        dem_resolution: Resolution of the DEM
    """
    SPEED_OF_LIGHT = 299792458.0
    SPACECRAFT_HEIGHT = 693000.0
    EARTH_RADIUS = 6337286.638938101

    parser = etree.XMLParser(encoding='utf-8', recover=True)
    reference_annotation_path = min((reference_safe_path / 'annotation').glob('s1*.xml'))
    ref_manifest_xml = etree.parse(str(reference_safe_path / 'manifest.safe'), parser)
    sec_manifest_xml = etree.parse(str(secondary_safe_path / 'manifest.safe'), parser)
    ref_annotation_xml = etree.parse(str(reference_annotation_path), parser)
    topsProc_xml = etree.parse(str(processing_path / 'topsProc.xml'), parser)
    topsApp_xml = etree.parse(str(processing_path / 'topsApp.xml'), parser)

    safe = '{http://www.esa.int/safe/sentinel-1.0}'
    s1 = '{http://www.esa.int/safe/sentinel-1.0/sentinel-1}'
    metadata_path = './/metadataObject[@ID="measurementOrbitReference"]//xmlData//'
    orbit_number_query = metadata_path + safe + 'orbitNumber'
    orbit_direction_query = metadata_path + safe + 'extension//' + s1 + 'pass'

    ref_orbit_number: str = ref_manifest_xml.find(orbit_number_query).text  # type: ignore[assignment, union-attr]
    ref_orbit_direction: str = ref_manifest_xml.find(orbit_direction_query).text  # type: ignore[assignment, union-attr]
    sec_orbit_number: str = sec_manifest_xml.find(orbit_number_query).text  # type: ignore[assignment, union-attr]
    sec_orbit_direction: str = sec_manifest_xml.find(orbit_direction_query).text  # type: ignore[assignment, union-attr]
    ref_heading = float(ref_annotation_xml.find('.//platformHeading').text)  # type: ignore[arg-type, union-attr]
    ref_time: str = ref_annotation_xml.find('.//productFirstLineUtcTime').text  # type: ignore[assignment, union-attr]
    slant_range_time = float(ref_annotation_xml.find('.//slantRangeTime').text)  # type: ignore[arg-type, union-attr]
    range_sampling_rate = float(ref_annotation_xml.find('.//rangeSamplingRate').text)  # type: ignore[arg-type, union-attr]
    number_samples = int(ref_annotation_xml.find('.//swathTiming/samplesPerBurst').text)  # type: ignore[arg-type, union-attr]
    unwrapper_type: str = topsApp_xml.find('.//property[@name="unwrapper name"]').text  # type: ignore[assignment, union-attr]
    phase_filter_strength: str = topsApp_xml.find('.//property[@name="filter strength"]').text  # type: ignore[assignment, union-attr]

    baseline_perp = get_baseline_perp(topsProc_xml)

    slant_range_near = float(slant_range_time) * SPEED_OF_LIGHT / 2
    range_pixel_spacing = SPEED_OF_LIGHT / (2 * range_sampling_rate)
    slant_range_far = slant_range_near + (number_samples - 1) * range_pixel_spacing
    slant_range_center = (slant_range_near + slant_range_far) / 2

    s = ref_time.split('T')[1].split(':')
    utc_time = ((int(s[0]) * 60 + int(s[1])) * 60) + float(s[2])

    parameter_file = ParameterFile(
        reference_granule=', '.join(reference_scenes),
        secondary_granule=', '.join(secondary_scenes),
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


def upload_product_to_s3(product_dir: Path, output_zip: Path, bucket: str, bucket_prefix: str) -> None:
    for browse in product_dir.glob('*.png'):
        create_thumbnail(browse, output_dir=product_dir)

    upload_file_to_s3(output_zip, bucket, bucket_prefix)

    for product_file in product_dir.iterdir():
        upload_file_to_s3(product_file, bucket, bucket_prefix)
