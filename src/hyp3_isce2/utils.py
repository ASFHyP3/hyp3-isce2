import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import isceobj  # type: ignore[import-not-found]
import numpy as np
from isceobj.Util.ImageUtil.ImageLib import loadImage  # type: ignore[import-not-found]
from iscesys.Component.ProductManager import ProductManager  # type: ignore[import-not-found]
from osgeo import gdal, osr


gdal.UseExceptions()


class GDALConfigManager:
    """Context manager for setting GDAL config options temporarily"""

    def __init__(self, **options):
        """Args:
        **options: GDAL Config `option=value` keyword arguments.
        """
        self.options = options.copy()
        self._previous_options = {}

    def __enter__(self):
        for key in self.options:
            self._previous_options[key] = gdal.GetConfigOption(key)

        for key, value in self.options.items():
            gdal.SetConfigOption(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self._previous_options.items():
            gdal.SetConfigOption(key, value)


@dataclass
class ParameterFile:
    reference_granule: str
    secondary_granule: str
    reference_orbit_direction: str
    reference_orbit_number: str
    secondary_orbit_direction: str
    secondary_orbit_number: str
    baseline: float
    utc_time: float
    heading: float
    spacecraft_height: float
    earth_radius_at_nadir: float
    slant_range_near: float
    slant_range_center: float
    slant_range_far: float
    range_looks: int
    azimuth_looks: int
    insar_phase_filter: bool
    phase_filter_parameter: float
    range_bandpass_filter: bool
    azimuth_bandpass_filter: bool
    dem_source: str
    dem_resolution: int
    unwrapping_type: str
    speckle_filter: bool
    water_mask: bool
    radar_n_lines: int | None = None
    radar_n_samples: int | None = None
    radar_first_valid_line: int | None = None
    radar_n_valid_lines: int | None = None
    radar_first_valid_sample: int | None = None
    radar_n_valid_samples: int | None = None
    multilook_azimuth_time_interval: float | None = None
    multilook_range_pixel_size: float | None = None
    radar_sensing_stop: datetime | None = None

    def __str__(self):
        output_strings = [
            f'Reference Granule: {self.reference_granule}\n',
            f'Secondary Granule: {self.secondary_granule}\n',
            f'Reference Pass Direction: {self.reference_orbit_direction}\n',
            f'Reference Orbit Number: {self.reference_orbit_number}\n',
            f'Secondary Pass Direction: {self.secondary_orbit_direction}\n',
            f'Secondary Orbit Number: {self.secondary_orbit_number}\n',
            f'Baseline: {self.baseline}\n',
            f'UTC time: {self.utc_time}\n',
            f'Heading: {self.heading}\n',
            f'Spacecraft height: {self.spacecraft_height}\n',
            f'Earth radius at nadir: {self.earth_radius_at_nadir}\n',
            f'Slant range near: {self.slant_range_near}\n',
            f'Slant range center: {self.slant_range_center}\n',
            f'Slant range far: {self.slant_range_far}\n',
            f'Range looks: {self.range_looks}\n',
            f'Azimuth looks: {self.azimuth_looks}\n',
            f'INSAR phase filter: {"yes" if self.insar_phase_filter else "no"}\n',
            f'Phase filter parameter: {self.phase_filter_parameter}\n',
            f'Range bandpass filter: {"yes" if self.range_bandpass_filter else "no"}\n',
            f'Azimuth bandpass filter: {"yes" if self.azimuth_bandpass_filter else "no"}\n',
            f'DEM source: {self.dem_source}\n',
            f'DEM resolution (m): {self.dem_resolution}\n',
            f'Unwrapping type: {self.unwrapping_type}\n',
            f'Speckle filter: {"yes" if self.speckle_filter else "no"}\n',
            f'Water mask: {"yes" if self.water_mask else "no"}\n',
        ]

        # TODO could use a more robust way to check if radar data is present
        if self.radar_n_lines:
            radar_data = [
                f'Radar n lines: {self.radar_n_lines}\n',
                f'Radar n samples: {self.radar_n_samples}\n',
                f'Radar first valid line: {self.radar_first_valid_line}\n',
                f'Radar n valid lines: {self.radar_n_valid_lines}\n',
                f'Radar first valid sample: {self.radar_first_valid_sample}\n',
                f'Radar n valid samples: {self.radar_n_valid_samples}\n',
                f'Multilook azimuth time interval: {self.multilook_azimuth_time_interval}\n',
                f'Multilook range pixel size: {self.multilook_range_pixel_size}\n',
                f'Radar sensing stop: {datetime.strftime(self.radar_sensing_stop, "%Y-%m-%dT%H:%M:%S.%f")}\n',  # type: ignore[arg-type]
            ]
            output_strings += radar_data

        return ''.join(output_strings)

    def __repr__(self):
        return self.__str__()

    def write(self, out_path: Path):
        out_path.write_text(self.__str__())


def utm_from_lon_lat(lon: float, lat: float) -> int:
    """Get the UTM zone EPSG code from a longitude and latitude.
    See https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
    for more details on UTM coordinate systems.

    Args:
        lon: Longitude
        lat: Latitude

    Returns:
        UTM zone EPSG code
    """
    hemisphere = 32600 if lat >= 0 else 32700
    zone = int(lon // 6 + 30) % 60 + 1
    return hemisphere + zone


def extent_from_geotransform(geotransform: tuple, x_size: int, y_size: int) -> tuple:
    """Get the extent and resolution of a GDAL dataset.

    Args:
        geotransform: GDAL geotransform.
        x_size: Number of pixels in the x direction.
        y_size: Number of pixels in the y direction.

    Returns:
        tuple: Extent of the dataset.
    """
    extent = (
        geotransform[0],
        geotransform[3],
        geotransform[0] + geotransform[1] * x_size,
        geotransform[3] + geotransform[5] * y_size,
    )
    return extent


def make_browse_image(input_tif: str, output_png: str) -> None:
    with GDALConfigManager(GDAL_PAM_ENABLED='NO'):
        try:
            stats = gdal.Info(input_tif, format='json', stats=True)['stac']['raster:bands'][0]['stats']
        except RuntimeError as error:
            if 'no valid pixels' in str(error):
                stats = {'minimum':0, 'maximum':0}
            else:
                raise
        gdal.Translate(
            destName=output_png,
            srcDS=input_tif,
            format='png',
            outputType=gdal.GDT_Byte,
            width=2048,
            strict=True,
            scaleParams=[[stats['minimum'], stats['maximum']]],
        )


def load_isce2_image(in_path) -> tuple[isceobj.Image, np.ndarray]:
    """Read an ISCE2 image file and return the image object and array.

    Args:
        in_path: The path to the image to resample (not the xml).

    Returns:
        image_obj: The ISCE2 image object.
        array: The image as a numpy array.
    """
    image_obj, _, _ = loadImage(in_path)
    array = np.fromfile(in_path, image_obj.toNumpyDataType())
    array = np.reshape(array, (-1, image_obj.width))
    if image_obj.bands > 1:
        if image_obj.imageType == 'bil':
            shape = (image_obj.bands, image_obj.length, image_obj.width)
            new_array = np.zeros(shape, dtype=image_obj.toNumpyDataType())
            for i in range(image_obj.bands):
                new_array[i, :, :] = array[i :: image_obj.bands]
            array = new_array.copy()
        else:
            raise NotImplementedError('Non-BIL reading is not implemented')
    return image_obj, array


def write_isce2_image(output_path: str, array: np.ndarray) -> None:
    """Write a numpy array as an ISCE2 image file.

    Args:
        output_path: The path to the output image file.
        array: The array to write to the file.
    """
    data_type_dic = {
        'float32': 'FLOAT',
        'float64': 'DOUBLE',
        'int32': 'INT',
        'complex64': 'CFLOAT',
        'int8': 'BYTE',
    }

    data_type = data_type_dic[str(array.dtype)]

    if array.ndim == 1:
        bands = 1
        length = 1
        width = array.shape[0]
    elif array.ndim == 2:
        bands = 1
        length, width = array.shape
    elif array.ndim == 3:
        bands, length, width = array.shape
    else:
        raise NotImplementedError('array with dimension larger than 3 is not implemented')

    image_obj = isceobj.createImage()
    image_obj.initImage(output_path, 'write', width, data_type, bands)
    image_obj.setLength(length)
    image_obj.setImageType('bil')
    image_obj.createImage()
    write_isce2_image_from_obj(image_obj, array)


def get_geotransform_from_dataset(dataset: isceobj.Image) -> tuple:
    """Get the geotransform from an ISCE2 image object.

    Args:
        dataset: The ISCE2 image object to get the geotransform from.

    Returns:
        tuple: The geotransform in GDAL Format: (startLon, deltaLon, 0, startLat, 0, deltaLat)
    """
    startLat = dataset.coord2.coordStart
    deltaLat = dataset.coord2.coordDelta
    startLon = dataset.coord1.coordStart
    deltaLon = dataset.coord1.coordDelta

    return (startLon, deltaLon, 0, startLat, 0, deltaLat)


def resample_to_radar(
    mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    geotransform: tuple,
    data_type: type,
    outshape: tuple[int, int],
) -> np.ndarray:
    """Resample a geographic image to radar coordinates using a nearest neighbor method.
    The latin and lonin images are used to map from geographic to radar coordinates.

    Args:
        mask: The array of the image to resample
        lat: The latitude array
        lon: The longitude array
        geotransform: The geotransform of the image to resample
        data_type: The data type of the image to resample
        outshape: The shape of the output image

    Returns:
        resampled_image: The resampled image array
    """
    start_lon, delta_lon, start_lat, delta_lat = (
        geotransform[0],
        geotransform[1],
        geotransform[3],
        geotransform[5],
    )

    lati = np.clip((((lat - start_lat) / delta_lat) + 0.5).astype(int), 0, mask.shape[0] - 1)
    loni = np.clip((((lon - start_lon) / delta_lon) + 0.5).astype(int), 0, mask.shape[1] - 1)
    resampled_image = (mask[lati, loni]).astype(data_type)
    resampled_image = np.reshape(resampled_image, outshape)
    return resampled_image


def resample_to_radar_io(image_to_resample: str, latin: str, lonin: str, output: str) -> None:
    """Resample a geographic image to radar coordinates using a nearest neighbor method.
    The latin and lonin images are used to map from geographic to radar coordinates.

    Args:
        image_to_resample: The path to the image to resample
        latin: The path to the latitude image
        lonin: The path to the longitude image
        output: The path to the output image
    """
    maskim, mask = load_isce2_image(image_to_resample)
    latim, lat = load_isce2_image(latin)
    _, lon = load_isce2_image(lonin)
    mask = np.reshape(mask, [maskim.coord2.coordSize, maskim.coord1.coordSize])
    geotransform = get_geotransform_from_dataset(maskim)
    cropped = resample_to_radar(
        mask=mask,
        lat=lat,
        lon=lon,
        geotransform=geotransform,
        data_type=maskim.toNumpyDataType(),
        outshape=(latim.coord2.coordSize, latim.coord1.coordSize),
    )

    write_isce2_image(output, array=cropped)


def isce2_copy(in_path: str, out_path: str):
    """Copy an ISCE2 image file and its metadata.

    Args:
        in_path: The path to the input image file (not the xml).
        out_path: The path to the output image file (not the xml).
    """
    image, _, _ = loadImage(in_path)
    clone = image.clone('write')
    clone.setFilename(out_path)
    clone.renderHdr()
    shutil.copy(in_path, out_path)


def image_math(image_a_path: str, image_b_path: str, out_path: str, expression: str):
    """Run ISCE2's imageMath.py on two images.

    Args:
        image_a_path: The path to the first image (not the xml).
        image_b_path: The path to the second image (not the xml).
        out_path: The path to the output image.
        expression: The expression to pass to imageMath.py.
    """
    cmd = [
        'imageMath.py',
        f'--a={image_a_path}',
        f'--b={image_b_path}',
        '-o',
        f'{out_path}',
        '--eval',
        expression,
    ]
    subprocess.run(cmd, check=True)


def load_product(xmlname: str):
    """Load an ISCE2 product from an xml file

    Args:
        xmlname: The path to the xml file

    Returns:
        The ISCE2 product
    """
    pm = ProductManager()
    pm.configure()
    obj = pm.loadProduct(xmlname)
    return obj


def write_isce2_image_from_obj(image_obj, array):
    """Write an ISCE2 image file.

    Args:
        image_obj: ISCE2 image object
        array: The array to write to the file.
    """
    image_obj.renderHdr()

    if image_obj.bands > 1:
        if image_obj.imageType == 'bil':
            shape = (image_obj.length * image_obj.bands, image_obj.width)
            new_array = np.zeros(shape, dtype=image_obj.toNumpyDataType())
            for i in range(image_obj.bands):
                new_array[i :: image_obj.bands] = array[i, :, :]
            array = new_array.copy()
        else:
            raise NotImplementedError('Non-BIL writing is not implemented')

    array.tofile(image_obj.filename)


def create_image(
    out_path: str,
    width: int | None = None,
    access_mode: str = 'read',
    image_subtype: str = 'default',
    action: str = 'create',
) -> isceobj.Image:
    """Create an ISCE2 image object from a set of parameters

    Args:
        out_path: The path to the output image
        width: The width of the image
        access_mode: The access mode of the image (read or write)
        image_subtype: The type of image to create
        action: What action to take:
            'create': create a new image object, but don't write metadata files, access_mode='write'
            'finalize': create a new image object based on existed binary file, and write metadata files,
             access_mode='read'
            'load': create an image object by loading an existing metadata file, access_mode='read'

    Returns:
        The ISCE2 image object
    """
    opts = {
        'ifg': (isceobj.createIntImage, 1, 'CFLOAT', 'cpx'),
        'cor': (isceobj.createImage, 1, 'FLOAT', 'cor'),
        'unw': (isceobj.Image.createUnwImage, 2, 'FLOAT', 'unw'),
        'conncomp': (isceobj.createImage, 1, 'BYTE', ''),
        'default': (isceobj.createImage, 1, 'FLOAT', ''),
    }

    create_func, bands, dtype, image_type = opts[image_subtype]
    image = create_func()
    if action == 'load':
        image.load(out_path + '.xml')
        image.setAccessMode('read')
        image.createImage()
        return image

    if width is None:
        raise ValueError('Width must be specified when the action is create or finalize')

    image.initImage(out_path, access_mode, width, dtype, bands)
    image.setImageType(image_type)
    if action == 'create':
        image.createImage()
    elif action == 'finalize':
        image.renderVRT()
        image.createImage()
        image.finalizeImage()
        image.renderHdr()
    return image


def read_product_metadata(meta_file_path: str) -> dict:
    """Read the HyP3-generated metadata file for a HyP3 product

    Args:
        meta_file_path: The path to the metadata file
    Returns:
        A dictionary of metadata values
    """
    hyp3_meta = {}
    with open(meta_file_path) as f:
        for line in f:
            key, *values = line.strip().replace(' ', '').split(':')
            value = ':'.join(values)
            hyp3_meta[key] = value
    return hyp3_meta


def get_projection(srs_wkt) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromWkt(srs_wkt)
    return srs.GetAttrValue('projcs')
