import copy
import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import asf_search
import numpy as np
import requests
from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1  # type: ignore[import-not-found]
from isceobj.TopsProc.runMergeBursts import multilook  # type: ignore[import-not-found]
from lxml import etree
from shapely import geometry

from hyp3_isce2.utils import load_isce2_image, load_product, write_isce2_image_from_obj


log = logging.getLogger(__name__)


URL = 'https://sentinel1-burst.asf.alaska.edu'


@dataclass
class BurstParams:
    """Parameters necessary to request a burst from the API."""

    granule: str
    swath: str
    polarization: str
    burst_number: int


@dataclass
class BurstPosition:
    """Parameters describing the position of a burst and its valid data."""

    n_lines: int
    n_samples: int
    first_valid_line: int
    n_valid_lines: int
    first_valid_sample: int
    n_valid_samples: int
    azimuth_time_interval: float
    range_pixel_size: float
    sensing_stop: datetime


class BurstMetadata:
    """Metadata for a burst."""

    def __init__(self, metadata: etree._Element, burst_params: BurstParams):
        self.swath: str = burst_params.swath
        self.polarization: str = burst_params.polarization
        self.burst_number: int = burst_params.burst_number

        self.safe_name: str = burst_params.granule
        self.manifest_name: str = 'manifest.safe'
        self.annotation_name: Path = Path()
        self.calibration_name: Path = Path()
        self.noise_name: Path = Path()

        self.manifest: etree._Element | None = metadata[0]
        self.annotation: etree._Element | None = None
        self.calibration: etree._Element | None = None
        self.noise: etree._Element | None = None

        metadata = metadata[1]

        names = [file.attrib['source_filename'] for file in metadata]
        lengths = [len(name.split('-')) for name in names]
        swaths = [name.split('-')[length - 8] for name, length in zip(names, lengths)]
        products = [x.tag for x in metadata]
        swaths_and_products = list(zip(swaths, products))

        files = {
            'product': 'annotation',
            'calibration': 'calibration',
            'noise': 'noise',
        }

        for name in files:
            elem = metadata[swaths_and_products.index((self.swath.lower(), name))]
            content = copy.deepcopy(elem.find('content'))
            if content:
                content.tag = 'product'
                setattr(self, files[name], content)
                setattr(self, f'{files[name]}_name', elem.attrib['source_filename'])
            else:
                raise ValueError(f'Could not find "content" attribute in {name}.')

        file_paths = [elements.attrib['href'] for elements in self.manifest.findall('.//fileLocation')]
        pattern = f'^./measurement/s1.*{self.swath.lower()}.*{self.polarization.lower()}.*.tiff$'
        self.measurement_name = [Path(path).name for path in file_paths if re.search(pattern, path)][0]

        orbit_direction = self.manifest.findtext('.//{*}pass')
        if orbit_direction:
            self.orbit_direction = orbit_direction.lower()
        else:
            raise ValueError(f'Could not find "pass" attribute in {name}.')


def create_burst_request_url(params: BurstParams, content_type: str) -> str:
    """Create a URL to request a burst from the API.

    Args:
        params: The burst search parameters.
        content_type: The content type of the burst to request.

    Returns:
        A URL to request a burst from the API.
    """
    filetypes = {'metadata': 'xml', 'geotiff': 'tiff'}
    extension = filetypes[content_type]
    url = f'{URL}/{params.granule}/{params.swath}/{params.polarization}/{params.burst_number}.{extension}'
    return url


def wait_for_extractor(response: requests.Response, sleep_time: int = 15) -> bool:
    """Wait for the burst extractor to finish processing a burst.

    Args:
        response: The response from the burst extractor.
        sleep_time: The number of seconds to wait between checking the status of the burst.

    Returns:
        True if the burst was successfully downloaded, False otherwise.
    """
    if response.status_code == 202:
        time.sleep(sleep_time)
        return False

    response.raise_for_status()
    return True


def download_from_extractor(asf_session: requests.Session, burst_params: BurstParams, content_type: str) -> bytes:
    """Download burst data from the extractor.

    Args:
        asf_session: A requests session with an ASF URS cookie.
        burst_params: The burst search parameters.
        content_type: The type of content to download (metadata or geotiff).

    Returns:
        The downloaded content.
    """
    burst_request_url = create_burst_request_url(burst_params, content_type=content_type)
    burst_request_cookies = {'asf-urs': asf_session.cookies['asf-urs']}

    for i in range(1, 11):
        log.info(f'Download attempt #{i} for {burst_request_url}')
        response = asf_session.get(url=burst_request_url, cookies=burst_request_cookies)
        downloaded = wait_for_extractor(response)
        if downloaded:
            break

    if not downloaded:
        raise RuntimeError('Download failed too many times')

    return response.content


def download_metadata(
    asf_session: requests.Session,
    burst_params: BurstParams,
    out_file: Path | str | None = None,
) -> etree._Element:
    """Download burst metadata.

    Args:
        asf_session: A requests session with an ASF URS cookie.
        burst_params: The burst search parameters.
        out_file: The path to save the metadata to (if desired).

    Returns:
        The metadata as an lxml.etree._Element object
    """
    content = download_from_extractor(asf_session, burst_params, 'metadata')
    metadata = etree.fromstring(content)

    if out_file:
        with open(out_file, 'wb') as f:
            f.write(content)

    return metadata


def download_burst(
    asf_session: requests.Session,
    burst_params: BurstParams,
    out_file: Path | str | None = None,
) -> Path:
    """Download a burst geotiff.

    Args:
        asf_session: A requests session with an ASF URS cookie.
        burst_params: The burst search parameters.
        out_file: The path to save the geotiff to (if desired).

    Returns:
        The path to the saved geotiff file.
    """
    content = download_from_extractor(asf_session, burst_params, 'geotiff')

    if not out_file:
        out_file = (
            f'{burst_params.granule}_{burst_params.swath}_{burst_params.polarization}_{burst_params.burst_number}.tiff'
        ).lower()

    with open(out_file, 'wb') as f:
        f.write(content)

    return Path(out_file)


def spoof_safe(burst: BurstMetadata, burst_tiff_path: Path, base_path: Path = Path()) -> Path:
    """Spoof a Sentinel-1 SAFE file for a burst.

    The created SAFE file will be saved to the base_path directory. The SAFE will have the following structure:
    SLC.SAFE/
    ├── manifest.safe
    ├── measurement/
    │   └── burst.tif
    └── annotation/
        ├── annotation.xml
        └── calibration/
            ├── calibration.xml
            └── noise.xml

    Args:
        burst: The burst metadata.
        burst_tiff_path: The path to the burst geotiff.
        base_path: The path to save the SAFE file to.

    Returns:
        The path to the saved SAFE file.
    """
    safe_path = base_path / f'{burst.safe_name}.SAFE'
    annotation_path = safe_path / 'annotation'
    calibration_path = safe_path / 'annotation' / 'calibration'
    measurement_path = safe_path / 'measurement'

    paths = [annotation_path, calibration_path, measurement_path]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    encoding = 'UTF-8'
    xml_dec = True

    filepaths: list[Path] = [
        annotation_path / burst.annotation_name,
        calibration_path / burst.calibration_name,
        calibration_path / burst.noise_name,
        safe_path / 'manifest.safe',
    ]

    elements: list[etree._Element | None] = [burst.annotation, burst.calibration, burst.noise, burst.manifest]

    for filepath, element in zip(filepaths, elements):
        assert isinstance(element, etree._Element)
        etree.ElementTree(element).write(filepath, encoding=encoding, xml_declaration=xml_dec)

    shutil.move(str(burst_tiff_path), str(measurement_path / burst.measurement_name))

    return safe_path


def get_isce2_burst_bbox(params: BurstParams, base_dir: Path | None = None) -> geometry.Polygon:
    """Get the bounding box of a Sentinel-1 burst using ISCE2.
    Using ISCE2 directly ensures that the bounding box is the same as the one used by ISCE2 for processing.

    Args:
        params: The burst parameters.
        base_dir: The directory containing the SAFE file.
            If base_dir is not set, it will default to the current working directory.

    returns:
        The bounding box of the burst as a shapely.geometry.Polygon object.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    s1_obj = Sentinel1()
    s1_obj.configure()
    s1_obj.polarization = params.polarization.lower()
    s1_obj.safe = [str(base_dir / f'{params.granule}.SAFE')]
    s1_obj.swathNumber = int(params.swath[-1])
    s1_obj.parse()
    snwe = s1_obj.product.bursts[params.burst_number].getBbox()

    # convert from south, north, west, east -> minx, miny, maxx, maxy
    bbox = geometry.box(snwe[2], snwe[0], snwe[3], snwe[1])
    return bbox


def get_region_of_interest(
    ref_bbox: geometry.Polygon, sec_bbox: geometry.Polygon, is_ascending: bool = True
) -> tuple[float, float, float, float]:
    """Get the region of interest for two bursts that will lead to single burst ISCE2 processing.

    For a descending orbit, the roi is in the lower left corner of the two bursts, and for an ascending orbit the roi is
    in the upper right corner.

    Args:
        ref_bbox: The reference burst's bounding box.
        sec_bbox: The secondary burst's bounding box.
        is_ascending: Whether the orbit is ascending or descending.

    Returns:
        The region of interest as a tuple of (minx, miny, maxx, maxy).
    """
    intersection = ref_bbox.intersection(sec_bbox)
    bounds = intersection.bounds

    x, y = (0, 1) if is_ascending else (2, 1)
    roi = geometry.Point(bounds[x], bounds[y]).buffer(0.005)
    return roi.bounds


def get_asf_session() -> requests.Session:
    """Get a requests session with an ASF URS cookie.

    requests will automatically use the netrc file:
    https://requests.readthedocs.io/en/latest/user/authentication/#netrc-authentication

    Returns:
        A requests session with an ASF URS cookie.
    """
    session = requests.Session()
    payload = {
        'response_type': 'code',
        'client_id': 'BO_n7nTIlMljdvU6kRRB3g',
        'redirect_uri': 'https://auth.asf.alaska.edu/login',
    }
    response = session.get('https://urs.earthdata.nasa.gov/oauth/authorize', params=payload)
    response.raise_for_status()
    return session


def download_bursts(param_list: list[BurstParams]) -> list[BurstMetadata]:
    """Download bursts in parallel and creates SAFE files.

    For each burst:
        1. Download metadata
        2. Download geotiff
        3. Create BurstMetadata object
        4. Create directory structure
        5. Write metadata
        6. Move geotiff to correct directory

    Args:
        param_list: An iterator of burst search parameters.

    Returns:
        A list of BurstMetadata objects.
    """
    with get_asf_session() as asf_session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            xml_futures = [executor.submit(download_metadata, asf_session, params) for params in param_list]
            tiff_futures = [executor.submit(download_burst, asf_session, params) for params in param_list]
            metadata_xmls = [future.result() for future in xml_futures]
            burst_paths = [future.result() for future in tiff_futures]

    bursts = []
    for params, metadata_xml, burst_path in zip(param_list, metadata_xmls, burst_paths):
        burst = BurstMetadata(metadata_xml, params)
        spoof_safe(burst, burst_path)
        bursts.append(burst)
    log.info('SAFEs created!')

    return bursts


def get_burst_params(scene_name: str) -> BurstParams:
    results = asf_search.search(product_list=[scene_name])

    if len(results) == 0:
        raise ValueError(f'ASF Search failed to find {scene_name}.')
    if len(results) > 1:
        raise ValueError(f'ASF Search found multiple results for {scene_name}.')

    return BurstParams(
        granule=results[0].umm['InputGranules'][0].split('-')[0],
        swath=results[0].properties['burst']['subswath'],
        polarization=results[0].properties['polarization'],
        burst_number=results[0].properties['burst']['burstIndex'],
    )


def _num_swath_pol(scene: str) -> str:
    parts = scene.split('_')
    num = parts[1]
    swath = parts[2]
    pol = parts[4]
    return '_'.join([num, swath, pol])


def validate_bursts(reference: str | list[str], secondary: str | list[str]) -> None:
    """Check whether the reference and secondary bursts are valid.

    Args:
        reference: Reference granule(s)
        secondary: Secondary granule(s)
    """
    if isinstance(reference, str):
        reference = [reference]

    if isinstance(secondary, str):
        secondary = [secondary]

    if len(reference) < 1 or len(secondary) < 1:
        raise ValueError('Must include at least 1 reference scene and 1 secondary scene')

    if len(reference) != len(secondary):
        raise ValueError(
            f'Must provide the same number of reference and secondary scenes, got {len(reference)} reference and {len(secondary)} secondary'
        )

    ref_num_swath_pol = [_num_swath_pol(g) for g in reference]
    sec_num_swath_pol = [_num_swath_pol(g) for g in secondary]

    for i in range(len(reference)):
        if ref_num_swath_pol[i] != sec_num_swath_pol[i]:
            raise ValueError(
                f'Number + swath + polarization identifier does not match for reference scene {reference[i]} and secondary scene {secondary[i]}'
            )

    pols = list(set(g.split('_')[4] for g in reference))

    if len(pols) > 1:
        raise ValueError(f'Scenes must have the same polarization. Polarizations present: {", ".join(sorted(pols))}')

    if pols[0] not in ['VV', 'HH']:
        raise ValueError(f'{pols[0]} polarization is not currently supported, only VV and HH')

    ref_dates = list(set(g.split('_')[3][:8] for g in reference))
    sec_dates = list(set(g.split('_')[3][:8] for g in secondary))

    if len(ref_dates) > 1:
        raise ValueError(f'Reference scenes must be from a single date. Dates present: {", ".join(sorted(ref_dates))}')

    if len(sec_dates) > 1:
        raise ValueError(f'Secondary scenes must be from a single date. Dates present: {", ".join(sorted(sec_dates))}')

    if ref_dates[0] >= sec_dates[0]:
        raise ValueError('Reference scenes must be older than secondary scenes')


def load_burst_position(swath_xml_path: str, burst_number: int) -> BurstPosition:
    """Get the tiff resolution and position parameters for a burst.

    Args:
        swath_xml_path: The path to the swath xml file.
        burst_number: The burst number.

    Returns:
        A BurstPosition object describing the burst.
    """
    product = load_product(swath_xml_path)
    burst_props = product.bursts[burst_number]

    pos = BurstPosition(
        n_lines=burst_props.numberOfLines,
        n_samples=burst_props.numberOfSamples,
        first_valid_line=burst_props.firstValidLine,
        n_valid_lines=burst_props.numValidLines,
        first_valid_sample=burst_props.firstValidSample,
        n_valid_samples=burst_props.numValidSamples,
        azimuth_time_interval=burst_props.azimuthTimeInterval,
        range_pixel_size=burst_props.rangePixelSize,
        sensing_stop=burst_props.sensingStop,
    )
    return pos


def evenize(length: int, first_valid: int, valid_length: int, looks: int) -> tuple[int, int, int]:
    """Get dimensions for an image that are integer multiples of looks.
    This applies to both the full image and the valid data region.
    Works with either the image's lines or samples.

    Args:
        length: The length of the image.
        first_valid: The first valid pixel of the image.
        valid_length: The length of the valid data region.
        looks: The number of looks.
    """
    # even_length must be a multiple of looks
    n_remove = length % looks
    even_length = length - n_remove

    # even_first_valid is the first multiple of looks after first_valid
    even_first_valid = int(np.ceil(first_valid / looks)) * looks

    # account for the shift introduced by the shift of first_valid
    n_first_valid_shift = even_first_valid - first_valid
    new_valid_length = valid_length - n_first_valid_shift

    # even_valid_length must be a multiple of looks
    n_valid_length_remove = new_valid_length % looks
    even_valid_length = new_valid_length - n_valid_length_remove

    if (even_first_valid + even_valid_length) > even_length:
        raise ValueError('The computed valid data region extends beyond the image bounds.')

    return even_length, even_first_valid, even_valid_length


def evenly_subset_position(position: BurstPosition, rg_looks, az_looks) -> BurstPosition:
    """Get the parameters necessary to multilook a burst using even dimensions.

    Multilooking using the generated parameters ensures that there is a clear link
    between pixels in the full resolution and multilooked pixel positions.

    Args:
        position: The BurstPosition object describing the burst.
        rg_looks: The number of range looks.
        az_looks: The number of azimuth looks.

    Returns:
        A BurstPosition object describing the burst.
    """
    even_n_samples, even_first_valid_sample, even_n_valid_samples = evenize(
        position.n_samples,
        position.first_valid_sample,
        position.n_valid_samples,
        rg_looks,
    )
    even_n_lines, even_first_valid_line, even_n_valid_lines = evenize(
        position.n_lines, position.first_valid_line, position.n_valid_lines, az_looks
    )
    n_lines_remove = position.n_lines - even_n_lines
    even_sensing_stop = position.sensing_stop - timedelta(seconds=position.azimuth_time_interval * n_lines_remove)

    clip_position = BurstPosition(
        n_lines=even_n_lines,
        n_samples=even_n_samples,
        first_valid_line=even_first_valid_line,
        n_valid_lines=even_n_valid_lines,
        first_valid_sample=even_first_valid_sample,
        n_valid_samples=even_n_valid_samples,
        azimuth_time_interval=position.azimuth_time_interval,
        range_pixel_size=position.range_pixel_size,
        sensing_stop=even_sensing_stop,
    )
    return clip_position


def multilook_position(position: BurstPosition, rg_looks: int, az_looks: int) -> BurstPosition:
    """Multilook a BurstPosition object.

    Args:
        position: The BurstPosition object to multilook.
        rg_looks: The number of range looks.
        az_looks: The number of azimuth looks.
    """
    return BurstPosition(
        n_lines=int(position.n_lines / az_looks),
        n_samples=int(position.n_samples / rg_looks),
        first_valid_line=int(position.first_valid_line / az_looks),
        n_valid_lines=int(position.n_valid_lines / az_looks),
        first_valid_sample=int(position.first_valid_sample / rg_looks),
        n_valid_samples=int(position.n_valid_samples / rg_looks),
        azimuth_time_interval=position.azimuth_time_interval * az_looks,
        range_pixel_size=position.range_pixel_size * rg_looks,
        sensing_stop=position.sensing_stop,
    )


def safely_multilook(
    in_file: str,
    position: BurstPosition,
    rg_looks: int,
    az_looks: int,
    subset_to_valid: bool = True,
) -> None:
    """Multilook an image, but only over a subset of the data whose dimensions are
    integer divisible by range/azimuth looks. Do the same for the valid data region.

    Args:
        in_file: The path to the input ISCE2-formatted image.
        position: The BurstPosition object describing the burst.
        rg_looks: The number of range looks.
        az_looks: The number of azimuth looks.
        subset_to_valid: Whether to subset the image to the valid data region specified by position.
    """
    image_obj, array = load_isce2_image(in_file)

    dtype = image_obj.toNumpyDataType()
    identity_value = np.identity(1, dtype=dtype)
    mask = np.zeros((position.n_lines, position.n_samples), dtype=dtype)

    if subset_to_valid:
        last_line = position.first_valid_line + position.n_valid_lines
        last_sample = position.first_valid_sample + position.n_valid_samples
        mask[
            position.first_valid_line : last_line,
            position.first_valid_sample : last_sample,
        ] = identity_value
    else:
        mask[:, :] = identity_value

    if len(array.shape) == 2:
        array = array[: position.n_lines, : position.n_samples].copy()
        array *= mask
    else:
        array = array[:, : position.n_lines, : position.n_samples].copy()
        for band in range(array.shape[0]):
            array[band, :, :] *= mask

    original_path = Path(image_obj.filename)
    clip_path = str(original_path.parent / f'{original_path.stem}.clip{original_path.suffix}')
    multilook_path = str(original_path.parent / f'{original_path.stem}.multilooked{original_path.suffix}')

    image_obj.setFilename(clip_path)
    image_obj.setWidth(position.n_samples)
    image_obj.setLength(position.n_lines)
    write_isce2_image_from_obj(image_obj, array)

    multilook(clip_path, multilook_path, alks=az_looks, rlks=rg_looks)


def multilook_radar_merge_inputs(
    swath_number: int, rg_looks: int, az_looks: int, base_dir: Path | None = None
) -> BurstPosition:
    """Multilook the radar datasets needed for post-generation product merging.

    Args:
        swath_number: The swath number.
        rg_looks: The number of range looks.
        az_looks: The number of azimuth looks.
        base_dir: The working directory. If not set, defaults to the current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    ifg_dir = base_dir / 'fine_interferogram'
    geom_dir = base_dir / 'geom_reference'

    swath = f'IW{swath_number}'
    position_params = load_burst_position(str(ifg_dir / f'{swath}.xml'), 0)
    even_position_params = evenly_subset_position(position_params, rg_looks, az_looks)
    safely_multilook(str(ifg_dir / swath / 'burst_01.int'), even_position_params, rg_looks, az_looks)

    for geom in ['lat_01.rdr', 'lon_01.rdr', 'los_01.rdr']:
        geom_path = str(geom_dir / swath / geom)
        safely_multilook(geom_path, even_position_params, rg_looks, az_looks, subset_to_valid=False)

    multilooked_params = multilook_position(even_position_params, rg_looks, az_looks)
    return multilooked_params
