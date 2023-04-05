import copy
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Iterator,
    List,
    Tuple,
    Union
)

import pandas as pd
import requests
from shapely import geometry


URL = 'https://sentinel1-burst.asf.alaska.edu'


@dataclass
class BurstParams:
    """Class that contains the parameters nessecary to request a burst from the API."""

    granule: str
    swath: str
    polarization: str
    burst_number: int


class BurstMetadata:
    """Class that contains the metadata for a burst."""

    def __init__(self, metadata: ET.Element, burst_params: BurstParams):
        self.safe_name = burst_params.granule
        self.swath = burst_params.swath
        self.polarization = burst_params.polarization
        self.burst_number = burst_params.burst_number
        self.manifest = metadata[0]
        self.manifest_name = 'manifest.safe'
        metadata = metadata[1]

        names = [x.attrib['source_filename'] for x in metadata]
        lengths = [len(x.split('-')) for x in names]
        swaths = [x.split('-')[y - 8] for x, y in zip(names, lengths)]
        products = [x.tag for x in metadata]
        swaths_and_products = list(zip(swaths, products))

        files = {'product': 'annotation', 'calibration': 'calibration', 'noise': 'noise'}
        for name in files:
            elem = metadata[swaths_and_products.index((self.swath.lower(), name))]
            content = copy.deepcopy(elem.find('content'))
            content.tag = 'product'
            setattr(self, files[name], content)
            setattr(self, f'{files[name]}_name', elem.attrib['source_filename'])

        file_paths = [x.attrib['href'] for x in self.manifest.findall('.//fileLocation')]
        pattern = f'^./measurement/s1.*{self.swath.lower()}.*{self.polarization.lower()}.*.tiff$'
        self.measurement_name = [Path(x).name for x in file_paths if re.search(pattern, x)][0]

        self.gcp_df = self.create_gcp_df()
        self.footprint = self.create_geometry(self.gcp_df)[0]
        self.orbit_direction = self.manifest.findtext('.//{*}pass').lower()

    @staticmethod
    def reformat_gcp(point: ET.Element) -> dict:
        """Reformat a burst geolocation grid point to a dictionary.

        Args:
            point: The geolocation grid point to reformat.

        Returns:
            A dictionary containing the geolocation grid point's line, pixel, latitude, longitude, and height.
        """
        attribs = ['line', 'pixel', 'latitude', 'longitude', 'height']
        values = {}
        for attrib in attribs:
            values[attrib] = float(point.find(attrib).text)
        return values

    def create_gcp_df(self) -> pd.DataFrame:
        """Create a dataframe of geolocation grid points.

        Returns:
            A dataframe containing the geolocation grid points for the burst.
        """
        points = self.annotation.findall('.//{*}geolocationGridPoint')
        gcp_df = pd.DataFrame([self.reformat_gcp(x) for x in points])
        gcp_df = gcp_df.sort_values(['line', 'pixel']).reset_index(drop=True)
        return gcp_df

    def create_geometry(self, gcp_df: pd.DataFrame) -> Tuple[geometry.Polygon, Tuple[float], Tuple[float]]:
        """Create a shapely polygon and centroid for the burst.

        Args:
            gcp_df: A dataframe containing the geolocation grid points for the burst.

        Returns:
            A tuple containing the shapely polygon, bounding box, and centroid for the burst.
        """
        burst_index = self.burst_number - 1
        lines = int(self.annotation.findtext('.//{*}linesPerBurst'))
        first_line = gcp_df.loc[gcp_df['line'] == burst_index * lines, ['longitude', 'latitude']]
        second_line = gcp_df.loc[gcp_df['line'] == (burst_index + 1) * lines, ['longitude', 'latitude']]
        x1 = first_line['longitude'].tolist()
        y1 = first_line['latitude'].tolist()
        x2 = second_line['longitude'].tolist()
        y2 = second_line['latitude'].tolist()
        x2.reverse()
        y2.reverse()
        x = x1 + x2
        y = y1 + y2
        footprint = geometry.Polygon(zip(x, y))
        centroid = tuple([x[0] for x in footprint.centroid.xy])
        return footprint, footprint.bounds, centroid


def create_burst_request_url(params: BurstParams, content: str) -> str:
    """Create a URL to request a burst from the API.

    Args:
        params: The burst search parameters.

    Returns:
        A URL to request a burst from the API.
    """
    filetypes = {'metadata': 'xml', 'geotiff': 'tiff'}
    extension = filetypes[content]
    burst_number_zero_indexed = params.burst_number - 1
    url = f'{URL}/{params.granule}/{params.swath}/{params.polarization}/{burst_number_zero_indexed}.{extension}'
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
        time.sleep(15)
        return False

    response.raise_for_status()
    return True


def download_from_extractor(asf_session: requests.Session, burst_params: BurstParams, content: str) -> bytes:
    """Download burst data from the extractor.

    Args:
        asf_session: A requests session with an ASF URS cookie.
        burst_params: The burst search parameters.
        content: The type of content to download (metadata or geotiff).

    Returns:
        The downloaded content.
    """
    burst_request = {
        'url': create_burst_request_url(burst_params, content=content),
        'cookies': {'asf-urs': asf_session.cookies['asf-urs']},
    }

    for i in range(1, 11):
        print(f'Download attempt #{i}')
        response = asf_session.get(**burst_request)
        downloaded = wait_for_extractor(response)
        if downloaded:
            break

    if not downloaded:
        raise RuntimeError('Download failed too many times')

    return response.content


def download_metadata(asf_session: requests.Session, burst_params: BurstParams, out_file: Union[Path, str] = None):
    """Download burst metadata.

    Args:
        asf_session: A requests session with an ASF URS cookie.
        burst_params: The burst search parameters.
        out_file: The path to save the metadata to (if desired).

    Returns:
        The metadata as an ElementTree object or the path to the saved metadata file.
    """
    content = download_from_extractor(asf_session, burst_params, 'metadata')
    metadata = ET.fromstring(content)

    if not out_file:
        return metadata

    with open(out_file, 'wb') as f:
        f.write(content)

    return str(out_file)


def download_burst(asf_session: requests.Session, burst_params: BurstParams, out_file: Union[Path, str]) -> str:
    """Download a burst geotiff.

    Args:
        asf_session: A requests session with an ASF URS cookie.
        burst_params: The burst search parameters.
        out_file: The path to save the geotiff to (if desired).

    Returns:
        The path to the saved geotiff file.
    """
    content = download_from_extractor(asf_session, burst_params, 'geotiff')

    with open(out_file, 'wb') as f:
        f.write(content)

    return str(out_file)


def spoof_safe(asf_session: requests.Session, burst: BurstMetadata, base_path: Path = Path('.')) -> Path:
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
        asf_session: A requests session with an ASF URS cookie.
        burst: The burst metadata.
        base_path: The path to save the SAFE file to.

    Returns:
        The path to the saved SAFE file.
    """
    safe_path = base_path / f'{burst.safe_name}.SAFE'
    annotation_path = safe_path / 'annotation'
    calibration_path = safe_path / 'annotation' / 'calibration'
    measurement_path = safe_path / 'measurement'
    paths = [annotation_path, calibration_path, measurement_path]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

    et_args = {'encoding': 'UTF-8', 'xml_declaration': True}

    ET.ElementTree(burst.annotation).write(annotation_path / burst.annotation_name, **et_args)
    ET.ElementTree(burst.calibration).write(calibration_path / burst.calibration_name, **et_args)
    ET.ElementTree(burst.noise).write(calibration_path / burst.noise_name, **et_args)
    ET.ElementTree(burst.manifest).write(safe_path / 'manifest.safe', **et_args)

    burst_params = BurstParams(burst.safe_name, burst.swath, burst.polarization, burst.burst_number)
    download_burst(asf_session, burst_params, measurement_path / burst.measurement_name)

    return safe_path


def get_region_of_interest(poly1: geometry.Polygon, poly2: geometry.Polygon, is_ascending: bool = True) -> Tuple[float]:
    """Get the region of interest for two bursts that will lead to single burst ISCE2 processing.

    For a descending orbit, the roi is in the lower left corner of the two bursts, and for an ascending orbit the roi is
    in the upper right corner.

    Args:
        poly1: The first burst's footprint.
        poly2: The second burst's footprint.
        is_ascending: Whether the orbit is ascending or descending.

    Returns:
        The region of interest as a tuple of (minx, miny, maxx, maxy).
    """
    bbox1 = geometry.box(*poly1.bounds)
    bbox2 = geometry.box(*poly2.bounds)
    intersection = bbox1.intersection(bbox2)
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


def download_bursts(param_list: Iterator[BurstParams]) -> List[BurstMetadata]:
    """Download bursts and create SAFE files.

    For each burst:
        1. Download metadata
        2. Create BurstMetadata object
        3. Create directory structure
        4. Write metadata
        5. Download and write geotiff

    Args:
        param_list: An iterator of burst search parameters.

    Returns:
        A list of BurstMetadata objects.
    """
    bursts = []
    with get_asf_session() as asf_session:
        for i, params in enumerate(param_list):
            print(f'Creating SAFE {i+1}...')
            metadata_xml = download_metadata(asf_session, params)
            burst = BurstMetadata(metadata_xml, params)
            spoof_safe(asf_session, burst)
            bursts.append(burst)

    print('SAFEs created!')

    return bursts
