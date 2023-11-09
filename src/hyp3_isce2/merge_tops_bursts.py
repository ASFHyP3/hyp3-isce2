import argparse
import copy
import datetime
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from secrets import token_hex
from shutil import make_archive
from tempfile import TemporaryDirectory
from typing import Iterable, Tuple

import asf_search
from hyp3lib.util import string_is_true
import lxml.etree as ET
import numpy as np
from osgeo import gdal
from shapely import geometry

import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
from isceobj.TopsProc.runMergeBursts import mergeBox, mergeBursts2
from isceobj.TopsProc.runIon import maskUnwrap
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import Orbit
from iscesys.Component import createTraitSeq
from iscesys.Component.ProductManager import ProductManager
from mroipac.icu.Icu import Icu
from mroipac.filter.Filter import Filter
from stdproc.rectify.geocode.Geocodable import Geocodable
from zerodop.geozero import createGeozero

from hyp3_isce2.dem import download_dem_for_isce2
import hyp3_isce2.burst as burst_utils
from hyp3_isce2.utils import make_browse_image, image_math, resample_to_radar_io
from hyp3_isce2.water_mask import create_water_mask

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO, force=True)
log = logging.getLogger(__name__)

from hyp3_isce2.insar_tops_burst import get_pixel_size, translate_outputs  # noqa


BURST_IFG_DIR = 'fine_interferogram'
BURST_GEOM_DIR = 'geom_reference'
FILT_WRP_IFG_NAME = 'filt_topophase.flat'
UNW_IFG_NAME = 'filt_topophase.unw'
WRP_IFG_NAME = 'topophase.flat'
COH_NAME = 'phsig.cor'
LOS_NAME = 'los.rdr'
LAT_NAME = 'lat.rdr'
LON_NAME = 'lon.rdr'
CCOM_NAME = 'filt_topophase.unw.conncomp'
GEOCODE_LIST = [
    UNW_IFG_NAME,
    COH_NAME,
    LOS_NAME,
    FILT_WRP_IFG_NAME,
    CCOM_NAME,
]


@dataclass
class BurstProduct:
    """A dataclass to hold burst metadata"""

    granule: str
    reference_date: datetime.datetime
    secondary_date: datetime.datetime
    burst_id: int
    swath: str
    polarization: str
    burst_number: int
    product_path: Path
    n_lines: int
    n_samples: int
    range_looks: int
    azimuth_looks: int
    first_valid_line: int
    n_valid_lines: int
    first_valid_sample: int
    n_valid_samples: int
    az_time_interval: float
    rg_pixel_size: float
    start_utc: datetime.datetime
    stop_utc: datetime.datetime
    relative_orbit: int
    isce2_burst_number: int = field(default=None)

    def to_burst_params(self):
        """Convert to a burst_utils.BurstParams object"""
        return burst_utils.BurstParams(self.granule, self.swath, self.polarization, self.burst_number)


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


def get_burst_metadata(product_paths: Iterable[Path]) -> Iterable[BurstProduct]:
    """Create a list of BurstProduct objects from a set of burst product paths

    Args:
        product_paths: A list of paths to unzipped burst product directories

    Returns:
        A list of BurstProduct objects representing the burst metadata
    """
    meta_file_paths = [path / f'{path.name}.txt' for path in product_paths]
    metas = [read_product_metadata(path) for path in meta_file_paths]

    # TODO why does asf_search not return values in order?
    results = [asf_search.granule_search(item['ReferenceGranule'])[0] for item in metas]

    relative_orbits = [result.properties['pathNumber'] for result in results]
    granules = [Path(result.properties['url']).parts[2] for result in results]
    pattern = '%Y%m%dT%H%M%S'
    reference_granules = [datetime.datetime.strptime(item['ReferenceGranule'].split('_')[3], pattern) for item in metas]
    secondary_granules = [datetime.datetime.strptime(item['SecondaryGranule'].split('_')[3], pattern) for item in metas]
    swaths = [result.properties['burst']['subswath'] for result in results]
    burst_ids = [result.properties['burst']['relativeBurstID'] for result in results]
    burst_indexes = [result.properties['burst']['burstIndex'] for result in results]
    polarization = [result.properties['polarization'] for result in results]
    start_utc = [
        datetime.datetime.strptime(result.properties['startTime'], '%Y-%m-%dT%H:%M:%S.%fZ') for result in results
    ]
    relative_orbits = [result.properties['pathNumber'] for result in results]
    n_lines = [meta['Radarnlines'] for meta in metas]
    n_samples = [meta['Radarnsamples'] for meta in metas]
    range_looks = [int(meta['Rangelooks']) for meta in metas]
    azimuth_looks = [int(meta['Azimuthlooks']) for meta in metas]
    first_valid_line = [meta['Radarfirstvalidline'] for meta in metas]
    n_valid_lines = [meta['Radarnvalidlines'] for meta in metas]
    first_valid_sample = [meta['Radarfirstvalidsample'] for meta in metas]
    n_valid_samples = [meta['Radarnvalidsamples'] for meta in metas]
    az_time_interval = [meta['Multilookazimuthtimeinterval'] for meta in metas]
    rg_pixel_size = [meta['Multilookrangepixelsize'] for meta in metas]
    stop_utc = [datetime.datetime.strptime(meta['Radarsensingstop'], '%Y-%m-%dT%H:%M:%S.%f') for meta in metas]
    products = []
    for i in range(len(granules)):
        product = BurstProduct(
            granules[i],
            reference_granules[i],
            secondary_granules[i],
            burst_ids[i],
            swaths[i],
            polarization[i],
            burst_indexes[i],
            product_paths[i],
            n_lines[i],
            n_samples[i],
            range_looks[i],
            azimuth_looks[i],
            first_valid_line[i],
            n_valid_lines[i],
            first_valid_sample[i],
            n_valid_samples[i],
            az_time_interval[i],
            rg_pixel_size[i],
            start_utc[i],
            stop_utc[i],
            relative_orbits[i],
        )
        products.append(product)
    return products


def download_annotation_xmls(params: Iterable[burst_utils.BurstParams]) -> None:
    """Download annotation xmls for a set of burst parameters to a directory
    named 'annotation' in the current working directory

    Args:
        params: A list of burst_utils.BurstParams objects
    """
    polarization = list(set([param.polarization for param in params]))
    if len(polarization) > 1:
        raise ValueError('Only one polarization can be used')
    polarization = polarization[0]
    annotation_dir = Path('annotation')
    annotation_dir.mkdir(exist_ok=True)
    manifest_dir = Path('manifest')
    manifest_dir.mkdir(exist_ok=True)

    granules = list(set([param.granule for param in params]))
    download_params = []
    metadata_params = []
    for granule in granules:
        download_params.append(burst_utils.BurstParams(granule, 'IW1', polarization, 0))
        swaths = list(set([param.swath for param in params if param.granule == granule]))
        metadata_params += [burst_utils.BurstParams(granule, swath, polarization, 0) for swath in swaths]

    with burst_utils.get_asf_session() as asf_session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            xml_futures = [
                executor.submit(burst_utils.download_metadata, asf_session, param) for param in download_params
            ]
            metadata_xmls = {param.granule: future.result() for param, future in zip(download_params, xml_futures)}

    et_args = {'encoding': 'UTF-8', 'xml_declaration': True}
    for param in metadata_params:
        metadata_xml = metadata_xmls[param.granule]
        burst_metadata = burst_utils.BurstMetadata(metadata_xml, param)
        ET.ElementTree(burst_metadata.annotation).write(annotation_dir / burst_metadata.annotation_name, **et_args)
        ET.ElementTree(burst_metadata.manifest).write(manifest_dir / f'{burst_metadata.safe_name}.xml', **et_args)


def get_scene_roi(s1_obj_bursts: Iterable[isceobj.Sensor.TOPS.BurstSLC.BurstSLC]) -> Tuple:
    """Get the bounding box of a set of ISCE2 Sentinel1 burst objects

    Args:
        s1_obj_bursts: A list of ISCE2 Sentinel1 burst objects

    Returns:
        A tuple of (west, south, east, north) bounding box coordinates
    """
    for i, isce_burst in enumerate(s1_obj_bursts):
        snwe = isce_burst.getBbox()
        bbox = geometry.box(snwe[2], snwe[0], snwe[3], snwe[1])
        if i == 0:
            overall_bbox = bbox
        else:
            overall_bbox = overall_bbox.union(bbox)
    return overall_bbox.bounds


class Sentinel1BurstSelect(Sentinel1):
    """A Modified version of the ISCE2 Sentinel1 class that allows for subsetting of bursts"""

    def select_bursts(self, start_utcs: Iterable[datetime.datetime]) -> None:
        """Subset the burst list to only include bursts with start times in start_utcs

        Args:
            start_utcs: A list of burst start times to subset the burst list to
        """
        cropList = createTraitSeq('burst')
        tiffList = []
        eapList = []

        print('Number of Bursts before cropping: ', len(self.product.bursts))
        for start_utc in start_utcs:
            start_utc = start_utc.replace(microsecond=0)
            match = [
                x for x in enumerate(self.product.bursts) if x[1].burstStartUTC.replace(microsecond=0) == start_utc
            ]

            if not len(match) > 0:
                raise ValueError(f'No match for burst at time {start_utc} found.')
            ind, isce2_burst = match[0]

            cropList.append(isce2_burst)
            if len(self._tiffSrc):
                tiffList.append(self._tiffSrc[ind])
            eapList.append(self._elevationAngleVsTau[ind])

        # Actual cropping
        self.product.bursts = cropList
        self.product.numberOfBursts = len(self.product.bursts)

        self._tiffSrc = tiffList
        self._elevationAngleVsTau = eapList
        print('Number of Bursts after cropping: ', len(self.product.bursts))

    def update_burst_properties(self, products: Iterable[BurstProduct]) -> None:
        """Update burst properties based on the burst metadata and previous subset operations

        Args:
            products: A list of BurstProduct objects
        """
        width = self._burstWidth
        length = self._burstLength
        if width is None:
            width = self.product.bursts[0].numberOfSamples
        if length is None:
            length = self.product.bursts[0].numberOfLines

        for index, burst in enumerate(self.product.bursts):
            product = products[index]
            burst.firstValidLine = product.first_valid_line
            burst.numValidLines = product.n_valid_lines
            burst.firstValidSample = product.first_valid_sample
            burst.numValidSamples = product.n_valid_samples

            outfile = os.path.join(self.output, 'burst_%02d' % (index + 1) + '.slc')
            slcImage = isceobj.createSlcImage()
            slcImage.setByteOrder('l')
            slcImage.setFilename(outfile)
            slcImage.setAccessMode('read')
            slcImage.setWidth(width)
            slcImage.setLength(length)
            slcImage.setXmin(0)
            slcImage.setXmax(width)
            burst.image = slcImage
            burst.numberOfSamples = width
            burst.numberOfLines = length

            print('Updating burst number from {0} to {1}'.format(burst.burstNumber, index + 1))
            burst.burstNumber = index + 1

    def write_xml(self) -> None:
        """Write the product xml to the directory specified by self.output"""
        pm = ProductManager()
        pm.configure()

        outxml = self.output
        if outxml.endswith('/'):
            outxml = outxml[:-1]
        pm.dumpProduct(self.product, os.path.join(outxml + '.xml'))


def create_swath_objects(
    swath: int, products: Iterable[BurstProduct], polarization: str = 'VV', outdir: str = BURST_IFG_DIR
) -> Tuple[Iterable[BurstProduct], Sentinel1BurstSelect]:
    """Create an ISCE2 Sentinel1 instance for a set of burst products, and write the xml.
    Also updates the BurstProduct objects with the ISCE2 burst number.

    Args:
        swath: The swath number of the burst products
        products: A list of BurstProduct objects to create the ISCE2 Sentinel1 instance for
        polarization: The polarization of the burst products
        outdir: The directory to write the xml to

    Returns:
        A tuple of the updated BurstProduct objects and the ISCE2 Sentinel1 instance
    """
    swaths_in_products = list(set([int(product.swath[2:3]) for product in products]))
    if len(swaths_in_products) > 1 or swaths_in_products[0] != swath:
        raise ValueError(f'Products provided are not all in swath {swath}')

    annotation_xmls = [str(path) for path in Path('annotation').glob(f's1?-??{swath}-slc-{polarization.lower()}*')]
    if len(annotation_xmls) == 0:
        raise ValueError(
            f'No annotation files for swath {swath} and polarization {polarization} found in annotation directory'
        )
    manifest_xmls = [str(path) for path in Path('manifest').glob('S1*.xml')]

    Path(outdir).mkdir(exist_ok=True)
    obj = Sentinel1BurstSelect()
    obj.configure()
    obj.xml = annotation_xmls
    obj.tiff = ['' for _ in range(len(annotation_xmls))]
    obj.manifest = manifest_xmls
    obj.swath = swath
    obj.polarization = polarization.lower()
    obj.output = os.path.join(outdir, 'IW{0}'.format(swath))
    obj.parse()

    products = sorted(products, key=lambda x: x.start_utc)
    obj.select_bursts([b.start_utc for b in products])
    obj.update_burst_properties(products)
    obj.write_xml()

    for product, burst_obj in zip(products, obj.product.bursts):
        product.isce2_burst_number = burst_obj.burstNumber
    return products, obj


def modify_for_multilook(
    burst_products: Iterable[BurstProduct], swath_obj: Sentinel1BurstSelect, outdir: str = BURST_IFG_DIR
) -> None:
    """Modify a Sentinel1 instance so that it is compatible with previously multilooked burst products

    Args:
        burst_products: A list of BurstProduct objects containing the needed metadata
        swath_obj: A Sentinel1BurstSelect (or Sentinel1) instance representing the parent swath
        outdir: The directory to write the xml to
    """
    multilook_swath_obj = copy.deepcopy(swath_obj)
    multilook_swath_obj.output = os.path.join(outdir, 'IW{0}_multilooked'.format(multilook_swath_obj.swath))
    for new_metadata, burst_obj in zip(burst_products, multilook_swath_obj.product.bursts):
        burst_obj.numberOfSamples = new_metadata.n_samples
        burst_obj.numberOfLines = new_metadata.n_lines
        burst_obj.firstValidSample = new_metadata.first_valid_sample
        burst_obj.numValidSamples = new_metadata.n_valid_samples
        burst_obj.firstValidLine = new_metadata.first_valid_line
        burst_obj.numValidLines = new_metadata.n_valid_lines
        burst_obj.sensingStop = new_metadata.stop_utc
        burst_obj.azimuthTimeInterval = new_metadata.az_time_interval
        burst_obj.rangePixelSize = new_metadata.rg_pixel_size
    multilook_swath_obj.write_xml()


def download_dem(s1_objs: Iterable[Sentinel1BurstSelect]):
    """Download the DEM for the region covered in a set of ISCE2 Sentinel1 instances

    Args:
        s1_objs: A list of Sentinel1BurstSelect instances
    """
    burst_objs = []
    for s1_obj in s1_objs:
        burst_objs += s1_obj.product.bursts
    dem_roi = get_scene_roi(burst_objs)
    download_dem_for_isce2(dem_roi, dem_name='glo_30', dem_dir=Path.cwd(), buffer=0, resample_20m=False)


def translate_image(in_path: str, out_path: str, width: int, image_type: str) -> None:
    """Translate a HyP3 burst product image to an ISCE2 compatible image

    Args:
        in_path: The path to the input image
        out_path: The path to the output image
        width: The width of the image
        image_type: The type of image to translate can be one of 'int', 'los', 'lat', or 'lon'
    """
    if image_type in 'int':
        out_img = isceobj.createIntImage()
        n_bands = 1
        out_img.initImage(out_path, 'read', width, bands=n_bands)
    elif image_type in ['lat', 'lon']:
        n_bands = 1
        out_img = isceobj.createImage()
        out_img.initImage(out_path, 'read', width, 'DOUBLE', bands=n_bands)
    elif image_type == 'los':
        out_img = isceobj.createImage()
        n_bands = 2
        out_img.initImage(out_path, 'read', width, 'FLOAT', bands=n_bands, scheme='BIL')
    else:
        raise NotImplementedError(f'{image_type} is not a valid format')

    with TemporaryDirectory() as tmpdir:
        out_tmp_path = str(Path(tmpdir) / Path(out_path).name)
        gdal.Translate(
            out_tmp_path,
            in_path,
            bandList=[n + 1 for n in range(n_bands)],
            format='ENVI',
            creationOptions=['INTERLEAVE=BIL'],
        )
        shutil.copy(out_tmp_path, out_path)
    out_img.renderHdr()


def spoof_isce2_setup(burst_products: Iterable[BurstProduct], s1_obj: Sentinel1BurstSelect) -> None:
    """For a set of ASF burst products, create spoofed geom_reference and fine_interferogram directories
    that are in the state they would be in after running topsApp.py from the 'startup' step to the 'burstifg' step.

    Args:
        burst_products: A list of BurstProduct objects
        s1_obj: An ISCE2 Sentinel1 instance representing the parent swath
    """
    ifg_dir = Path('fine_interferogram')
    ifg_dir.mkdir(exist_ok=True)

    geom_dir = Path('geom_reference')
    geom_dir.mkdir(exist_ok=True)

    swaths = list(set([product.swath for product in burst_products]))
    for swath in swaths:
        ifg_swath_path = ifg_dir / swath
        ifg_swath_path.mkdir(exist_ok=True)

        geom_swath_path = geom_dir / swath
        geom_swath_path.mkdir(exist_ok=True)

    file_types = {
        'int': 'wrapped_phase_rdr',
        'los': 'los_rdr',
        'lat': 'lat_rdr',
        'lon': 'lon_rdr',
    }
    for product in burst_products:
        for image_type in file_types:
            if image_type in ['int', 'cor']:
                dir = ifg_dir
                name = f'burst_{product.isce2_burst_number:02}.{image_type}'
            else:
                dir = geom_dir
                name = f'{image_type}_{product.isce2_burst_number:02}.rdr'
            in_path = str(product.product_path / f'{product.product_path.stem}_{file_types[image_type]}.tif')
            out_path = str(dir / product.swath / name)
            translate_image(in_path, out_path, product.n_samples, image_type)


def get_swath_list(indir: str) -> list:
    """Get the list of swaths from a directory of burst products

    Args:
        indir: The directory containing the burst products

    Returns:
        A list of swaths
    """
    swathList = []
    for x in [1, 2, 3]:
        swath_paths = os.path.join(indir, 'IW{0}'.format(x))
        if os.path.exists(swath_paths):
            swathList.append(x)

    return swathList


def load_product(xmlname: str) -> Sentinel1:
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


def get_merged_orbit(products: Iterable[Sentinel1]) -> Orbit:
    """Create a merged orbit from a set of ISCE2 Sentinel1 products

    Args:
        product: A list of ISCE2 Sentinel1 products

    Returns:
        The merged orbit
    """
    ###Create merged orbit
    orb = Orbit()
    orb.configure()

    burst = products[0].bursts[0]
    # Add first burst orbit to begin with
    for sv in burst.orbit:
        orb.addStateVector(sv)

    for pp in products:
        ##Add all state vectors
        for bb in pp.bursts:
            for sv in bb.orbit:
                if (sv.time < orb.minTime) or (sv.time > orb.maxTime):
                    orb.addStateVector(sv)

            bb.orbit = orb

    return orb


def open_image(in_path: str, image_subtype: str = None) -> isceobj.Image.Image:
    """Open an image as an ISCE2 image object

    Args:
        in_path: The path to the image
        image_subtype: The type of image to open

    Returns:
        The ISCE2 image object
    """
    if image_subtype == 'ifg':
        image = isceobj.createIntImage()
    else:
        image = isceobj.createImage()
    image.load(in_path + '.xml')
    image.setAccessMode('read')
    image.createImage()
    return image


def create_image(
    out_path: str, width: int, access_mode: str, image_subtype: str = 'default', finalize: bool = False
) -> isceobj.Image.Image:
    """Create an ISCE2 image object from a set of parameters

    Args:
        out_path: The path to the output image
        width: The width of the image
        access_mode: The access mode of the image (read or write)
        image_subtype: The type of image to create
        finalize: Whether or not to write xml and hdr files

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
    image.initImage(out_path, access_mode, width, dtype, bands)
    image.setImageType(image_type)

    if finalize:
        image.renderVRT()
        image.createImage()
        image.finalizeImage()
        image.renderHdr()
    else:
        image.createImage()

    return image


def merge_bursts(range_looks: int, azimuth_looks: int, mergedir: str = 'merged') -> None:
    """Merge burst products into a multi-swath product, and multilook

    Args:
        azimuth_looks: The number of azimuth looks
        range_looks: The number of range looks
        mergedir: The directory to write the merged product to
    """
    frames = []
    burstIndex = []
    swathList = get_swath_list(BURST_IFG_DIR)
    for swath in swathList:
        ifg = load_product(os.path.join(BURST_IFG_DIR, 'IW{0}_multilooked.xml'.format(swath)))
        minBurst = ifg.bursts[0].burstNumber - 1
        maxBurst = ifg.bursts[-1].burstNumber
        frames.append(ifg)
        burstIndex.append([int(swath), minBurst, maxBurst])

    box = mergeBox(frames)
    file_types = {
        'ifg': (BURST_IFG_DIR, 'burst_%02d.int', WRP_IFG_NAME),
        'los': (BURST_GEOM_DIR, 'los_%02d.rdr', LOS_NAME),
        'lat': (BURST_GEOM_DIR, 'lat_%02d.rdr', LAT_NAME),
        'lon': (BURST_GEOM_DIR, 'lon_%02d.rdr', LON_NAME),
    }

    for file_type in file_types:
        directory, file_pattern, name = file_types[file_type]
        burst_paths = os.path.join(directory, 'IW%d', file_pattern)
        out_path = os.path.join(mergedir, name)
        merged_path = out_path
        mergeBursts2(frames, burst_paths, burstIndex, box, merged_path, virtual=True, validOnly=True)
        with TemporaryDirectory() as tmpdir:
            out_tmp_path = str(Path(tmpdir) / Path(out_path).name)
            gdal.Translate(out_tmp_path, out_path + '.vrt', format='ENVI', creationOptions=['INTERLEAVE=BIL'])
            shutil.copy(out_tmp_path, out_path)


def goldstein_werner_filter(filter_strength: float = 0.6, mergedir: str = 'merged') -> None:
    """Apply the Goldstein-Werner filter to the merged interferogram.
    See https://doi.org/10.1029/1998GL900033 for method details.

    Args:
        filter_strength: The filter strength
        mergedir: The output directory containing the merged interferogram
    """
    ifg_image = open_image(os.path.join(mergedir, WRP_IFG_NAME), image_subtype='ifg')
    width_ifg = ifg_image.getWidth()

    filt_ifg_filename = os.path.join(mergedir, FILT_WRP_IFG_NAME)
    filt_image = create_image(filt_ifg_filename, width_ifg, 'write', image_subtype='ifg')

    filter_obj = Filter()
    filter_obj.wireInputPort(name='interferogram', object=ifg_image)
    filter_obj.wireOutputPort(name='filtered interferogram', object=filt_image)
    filter_obj.goldsteinWerner(alpha=filter_strength)

    ifg_image.finalizeImage()
    filt_image.finalizeImage()
    del filt_image

    filt_image = create_image(filt_ifg_filename, width_ifg, 'read', image_subtype='ifg')
    phsigImage = create_image(os.path.join(mergedir, COH_NAME), width_ifg, 'write', image_subtype='cor')

    icu_obj = Icu(name='topsapp_filter_icu')
    icu_obj.configure()
    icu_obj.unwrappingFlag = False
    icu_obj.useAmplitudeFlag = False
    icu_obj.icu(intImage=filt_image, phsigImage=phsigImage)

    filt_image.finalizeImage()
    phsigImage.finalizeImage()
    phsigImage.renderHdr()


def mask_coherence(out_name, mergedir='merged'):
    """Mask the coherence image with a water mask that has been resampled to radar coordinates

    Args:
        mergedir: The output directory containing the merged interferogram
    """
    input_files = ('water_mask', LAT_NAME, LON_NAME, 'water_mask.rdr', COH_NAME, out_name)
    mask_geo, lat, lon, mask_rdr, coh, masked_coh = [str(Path(mergedir) / name) for name in input_files]
    create_water_mask(input_image='full_res.dem.wgs84', output_image=mask_geo, gdal_format='ISCE')
    resample_to_radar_io(mask_geo, lat, lon, mask_rdr)
    image_math(coh, mask_rdr, masked_coh, 'a*b')


def snaphu_unwrap(
    range_looks: int,
    azimuth_looks: int,
    corrfile: str = None,
    mergedir='merged',
    cost_mode='DEFO',
    init_method='MST',
    defomax=4.0,
) -> None:
    """Unwrap the merged interferogram using SNAPHU

    Args:
        azimuth_looks: The number of azimuth looks
        range_looks: The number of range looks
        corrfile: The path to the coherence file, will default to the coherence file in mergedir
        mergedir: The output directory containing the merged interferogram
        cost_mode: The cost mode to use for SNAPHU
        init_method: The initialization method to use for SNAPHU
        defomax: The maximum deformation allowed for SNAPHU
    """
    if not corrfile:
        corrfile = os.path.join(mergedir, COH_NAME)
    wrap_name = os.path.join(mergedir, FILT_WRP_IFG_NAME)
    unwrap_name = os.path.join(mergedir, UNW_IFG_NAME)

    img = isceobj.createImage()
    img.load(wrap_name + '.xml')
    width = img.getWidth()

    swath = get_swath_list(BURST_IFG_DIR)[0]
    ifg = load_product(os.path.join(BURST_IFG_DIR, 'IW{0}_multilooked.xml'.format(swath)))
    wavelength = ifg.bursts[0].radarWavelength

    # tmid
    tstart = ifg.bursts[0].sensingStart
    tend = ifg.bursts[-1].sensingStop
    tmid = tstart + 0.5 * (tend - tstart)

    # Sometimes tmid may exceed the time span, so use mid burst instead
    burst_index = int(np.around(len(ifg.bursts) / 2))
    orbit = ifg.bursts[burst_index].orbit
    peg = orbit.interpolateOrbit(tmid, method='hermite')

    # Calculate parameters for SNAPHU
    ref_elp = Planet(pname='Earth').ellipsoid
    llh = ref_elp.xyz_to_llh(peg.getPosition())
    hdg = orbit.getENUHeading(tmid)
    ref_elp.setSCH(llh[0], llh[1], hdg)
    earth_radius = ref_elp.pegRadCur
    altitude = llh[2]
    azfact = 0.8
    rngfact = 0.8
    corrLooks = range_looks * azimuth_looks / (azfact * rngfact)
    maxComponents = 20

    snp = Snaphu()
    snp.setInitOnly(False)
    snp.setInput(wrap_name)
    snp.setOutput(unwrap_name)
    snp.setWidth(width)
    snp.setCostMode(cost_mode)
    snp.setEarthRadius(earth_radius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corrfile)
    snp.setInitMethod(init_method)
    snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(range_looks)
    snp.setAzimuthLooks(azimuth_looks)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()

    # Render XML
    create_image(unwrap_name, width, 'read', image_subtype='unw', finalize=True)
    # Check if connected components was created
    if snp.dumpConnectedComponents:
        create_image(unwrap_name + '.conncomp', width, 'read', image_subtype='conncomp', finalize=True)
        maskUnwrap(unwrap_name, wrap_name)


def geocode_products(
    range_looks: int, azimuth_looks: int, dem_path, mergedir='merged', to_be_geocoded=GEOCODE_LIST
) -> None:
    """Geocode a set of ISCE2 products

    Args:
        azimuth_looks: The number of azimuth looks
        range_looks: The number of range looks
        dem_path: The path to the DEM
        mergedir: The output directory containing the merged interferogram
        to_be_geocoded: A list of products to geocode
    """
    to_be_geocoded = [str(Path(mergedir) / file) for file in to_be_geocoded]
    swath_list = get_swath_list(BURST_IFG_DIR)

    frames = []
    for swath in swath_list:
        reference_product = load_product(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath)))
        frames.append(reference_product)

    orbit = get_merged_orbit(frames)

    bboxes = []
    for frame in frames:
        bboxes.append(frame.getBbox())

    snwe = [
        min([x[0] for x in bboxes]),
        max([x[1] for x in bboxes]),
        min([x[2] for x in bboxes]),
        max([x[3] for x in bboxes]),
    ]

    # Identify the 4 corners and dimensions
    top_swath = min(frames, key=lambda x: x.sensingStart)
    left_swath = min(frames, key=lambda x: x.startingRange)

    # Get required values from product
    burst = frames[0].bursts[0]
    t0 = top_swath.sensingStart
    dtaz = burst.azimuthTimeInterval
    r0 = left_swath.startingRange
    dr = burst.rangePixelSize
    wvl = burst.radarWavelength
    planet = Planet(pname='Earth')

    # Setup DEM
    demImage = isceobj.createDemImage()
    demImage.load(dem_path + '.xml')

    # Geocode one by one
    ge = Geocodable()
    for prod in to_be_geocoded:
        geo_obj = createGeozero()
        geo_obj.configure()

        geo_obj.snwe = snwe
        geo_obj.demCropFilename = os.path.join(mergedir, 'dem.crop')
        geo_obj.numberRangeLooks = range_looks
        geo_obj.numberAzimuthLooks = azimuth_looks
        geo_obj.lookSide = -1  # S1A is currently right looking only

        # create the instance of the input image and the appropriate geocode method
        inImage, method = ge.create(prod)
        geo_obj.method = method

        geo_obj.slantRangePixelSpacing = dr
        geo_obj.prf = 1.0 / dtaz
        geo_obj.orbit = orbit
        geo_obj.width = inImage.getWidth()
        geo_obj.length = inImage.getLength()
        geo_obj.dopplerCentroidCoeffs = [0.0]
        geo_obj.radarWavelength = wvl

        geo_obj.rangeFirstSample = r0 + ((range_looks - 1) / 2.0) * dr
        geo_obj.setSensingStart(t0 + datetime.timedelta(seconds=(((azimuth_looks - 1) / 2.0) * dtaz)))
        geo_obj.wireInputPort(name='dem', object=demImage)
        geo_obj.wireInputPort(name='planet', object=planet)
        geo_obj.wireInputPort(name='tobegeocoded', object=inImage)

        geo_obj.geocode()


def get_product_name(product_directory: Path, pixel_size: int) -> str:
    """Create a product name for a merged burst product. Follows the convention used by ASF burst products,
    but replaces the burst id with the relative orbit number, and removes the swath compenent with ''.

    Args:
        product_directory: The path to the directory containing the ASF burst product directories
        pixel_size: The pixel size of the product

    Returns:
        The merged product name
    """
    example_name_split = list(product_directory.glob('S1_??????_IW?_*'))[0].name.split('_')

    swath_number = get_swath_list(BURST_IFG_DIR)[0]
    insar_product = load_product(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath_number)))

    platform = example_name_split[0]
    relative_orbit = f'{insar_product.bursts[0].trackNumber:03}'
    swath_placeholder = ''
    reference_date = example_name_split[3]
    secondary_date = example_name_split[4]
    polarization = example_name_split[5]
    product_type = 'INT'
    pixel_spacing = str(int(pixel_size))
    product_id = token_hex(2).upper()

    return '_'.join(
        [
            platform,
            relative_orbit,
            swath_placeholder,
            reference_date,
            secondary_date,
            polarization,
            product_type + pixel_spacing,
            product_id,
        ]
    )


def make_parameter_file(
    out_path: Path,
    product_directory: Path,
    range_looks: int,
    azimuth_looks: int,
    filter_strength: float,
    dem_name: str = 'GLO_30',
    dem_resolution: int = 30,
):
    """Create a parameter file for the ASF merged burst product and write it to out_path

    Args:
        out_path: The path to write the parameter file to
        product_directory: The path to the directory containing the ASF burst product directories
        azimuth_looks: The number of azimuth looks
        filter_strength: The Goldstein-Werner filter strength
        range_looks: The number of range looks
        dem_name: The name of the source DEM
        dem_resolution: The resolution of the source DEM
    """
    SPACECRAFT_HEIGHT = 693000.0
    EARTH_RADIUS = 6337286.638938101

    product_paths = list(product_directory.glob('S1_??????_IW?_*'))
    meta_file_paths = [path / f'{path.name}.txt' for path in product_paths]
    metas = [read_product_metadata(path) for path in meta_file_paths]
    reference_scenes = [meta['ReferenceGranule'] for meta in metas]
    secondary_scenes = [meta['SecondaryGranule'] for meta in metas]
    ref_orbit_number = metas[0]['ReferenceOrbitNumber']
    sec_orbit_number = metas[0]['SecondaryOrbitNumber']
    # TODO should we calculate this more accurately?
    baseline_perp = metas[0]['Baseline']

    swath_number = get_swath_list(BURST_IFG_DIR)[0]
    insar_product = load_product(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath_number)))

    orbit_direction = insar_product.bursts[0].passDirection
    ref_heading = insar_product.orbit.getHeading()
    ref_time = insar_product.bursts[0].sensingStart

    utc_time = (((ref_time.hour * 60) + ref_time.minute) * 60) + ref_time.second + (ref_time.microsecond * 10e-7)
    slant_range_near = insar_product.startingRange
    slant_range_center = insar_product.midRange
    slant_range_far = insar_product.farRange

    output_strings = [
        f'Reference Granule: {", ".join(reference_scenes)}\n',
        f'Secondary Granule: {", ".join(secondary_scenes)}\n',
        f'Reference Pass Direction: {orbit_direction}\n',
        f'Reference Orbit Number: {ref_orbit_number}\n',
        f'Secondary Pass Direction: {orbit_direction}\n',
        f'Secondary Orbit Number: {sec_orbit_number}\n',
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
        f'Phase filter parameter: {filter_strength}\n',
        'Range bandpass filter: no\n',
        'Azimuth bandpass filter: no\n',
        f'DEM source: {dem_name}\n',
        f'DEM resolution (m): {dem_resolution}\n',
        'Unwrapping type: snaphu_mcf\n',
        'Speckle filter: yes\n',
    ]

    output_string = ''.join(output_strings)

    with open(out_path.__str__(), 'w') as outfile:
        outfile.write(output_string)


def check_burst_group_validity(products) -> None:
    """Check that a set of burst products are valid for merging. This includes:
    All products have the same:
        - date
        - relative orbit
        - polarization
        - multilook
    All products must also be contiguous. This means:
        - A given swath has a continuous series of bursts
        - Neighboring swaths have at at most one burst separation

    This function will raise a ValueError if any of these conditions are not met.

    Args:
        products: A list of BurstProduct objects
    """
    reference_dates = set([product.reference_date.date() for product in products])
    secondary_dates = set([product.secondary_date.date() for product in products])
    polarizations = set([product.polarization for product in products])
    relative_orbits = set([product.relative_orbit for product in products])
    range_looks = [product.range_looks for product in products]
    azimuth_looks = [product.azimuth_looks for product in products]
    looks = set([f'{rgl}x{azl}' for rgl, azl in zip(range_looks, azimuth_looks)])

    sets = {
        'reference_date': reference_dates,
        'secondary_date': secondary_dates,
        'polarization': polarizations,
        'relative_orbit': relative_orbits,
        'looks': looks,
    }
    for key, value in sets.items():
        if len(value) > 1:
            key_name = key.replace('_', ' ')
            value_names = ", ".join([str(v) for v in value])
            raise ValueError(f'All products must have the same {key_name}. Found {value_names}.')

    swath_ids = {}
    for swath in set([product.swath for product in products]):
        swath_products = [product for product in products if product.swath == swath]
        swath_products.sort(key=lambda x: x.burst_id)
        ids = np.array([p.burst_id for p in swath_products])
        if not np.all(ids - ids.min() == np.arange(len(ids))):
            raise ValueError(f'Products for swath {swath} are not contiguous')
        swath_ids[swath] = ids

    for swath1, swath2 in combinations(swath_ids.keys(), 2):
        separations = np.concatenate([swath_ids[swath1] - elem for elem in swath_ids[swath2]])
        if separations.min() > 1:
            raise ValueError(f'Products from swaths {swath1} and {swath2} do not overlap')


def prepare_products(directory: Path) -> None:
    """Set up a directory for ISCE2-based burst merging using a set of ASF burst products.
    This includes:
    - Downloading annotation xml files
    - Downloading manifest xml files
    - Downloading a DEM
    - Spoofing the geom_reference and fine_interferogram directories

    Args:
        directory: The path to the directory containing the UNZIPPED ASF burst product directories
    """
    product_paths = list(directory.glob('S1_??????_IW?_*'))
    products = get_burst_metadata(product_paths)
    check_burst_group_validity(products)
    download_annotation_xmls([product.to_burst_params() for product in products])
    swaths = list(set([int(product.swath[2:3]) for product in products]))
    swath_objs = []
    for swath in swaths:
        swath_products = [product for product in products if int(product.swath[2:3]) == swath]
        swath_products, swath_obj = create_swath_objects(swath, swath_products)
        modify_for_multilook(swath_products, swath_obj)
        spoof_isce2_setup(swath_products, swath_obj)
        swath_objs.append(copy.deepcopy(swath_obj))
        del swath_obj

    download_dem(swath_objs)


def get_product_multilook(product_dir: Path) -> Tuple:
    """Get the multilook values for a set of ASF burst products.
    You should have already checked that all products have the same multilook,
    so you can just use the first product's values.

    Args:
        product_dir: The path to the directory containing the UNZIPPED ASF burst product directories

    Returns:
        The number of azimuth looks and range looks
    """
    product_path = list(product_dir.glob('S1_??????_IW?_*'))[0]
    metadata_path = product_path / f'{product_path.name}.txt'
    meta = read_product_metadata(metadata_path)
    return int(meta['Rangelooks']), int(meta['Azimuthlooks'])


def run_isce2_workflow(
    range_looks: int, azimuth_looks: int, mergedir='merged', filter_strength=0.5, apply_water_mask=False
) -> None:
    """Run the ISCE2 workflow for burst merging, filtering, unwrapping, and geocoding

    Args:
        azimuth_looks: The number of azimuth looks
        range_looks: The number of range looks
        mergedir: The output directory containing the merged interferogram
        filter_strength: The Goldstein-Werner filter strength
        apply_water_mask: Whether or not to apply a water body mask to the coherence file before unwrapping
    """
    Path(mergedir).mkdir(exist_ok=True)
    merge_bursts(range_looks, azimuth_looks, mergedir=mergedir)
    goldstein_werner_filter(filter_strength=filter_strength, mergedir=mergedir)
    if apply_water_mask:
        log.info('Water masking requested, downloading water mask')
        mask_coherence(f'masked.{COH_NAME}')
        corrfile = os.path.join(mergedir, f'masked.{COH_NAME}')
    else:
        corrfile = os.path.join(mergedir, COH_NAME)
    snaphu_unwrap(range_looks, azimuth_looks, corrfile=corrfile, mergedir=mergedir)
    geocode_products(range_looks, azimuth_looks, dem_path='full_res.dem.wgs84', mergedir=mergedir)


def package_output(product_directory: Path, looks: str, filter_strength: float, archive=False) -> None:
    """Package the output of the ISCE2 workflow into a the standard ASF burst product format

    Args:
        product_directory: The path to the directory containing the UNZIPPED ASF burst product directories
        looks: The number of looks [20x4, 10x2, 5x1]
        filter_strength: The Goldstein-Werner filter strength
        archive: Whether or not to create a zip archive of the output
    """
    pixel_size = get_pixel_size(looks)
    range_looks, azimuth_looks = [int(look) for look in looks.split('x')]

    product_name = get_product_name(product_directory, pixel_size)
    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    make_parameter_file(
        Path(f'{product_name}/{product_name}.txt'), product_directory, range_looks, azimuth_looks, filter_strength
    )
    translate_outputs(product_name, pixel_size=pixel_size, include_radar=False)
    unwrapped_phase = f'{product_name}/{product_name}_unw_phase.tif'
    make_browse_image(unwrapped_phase, f'{product_name}/{product_name}_unw_phase.png')
    if archive:
        make_archive(base_name=product_name, format='zip', base_dir=product_name)


def main():
    """HyP3 entrypoint for the TOPS burst merging workflow"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', type=str, help='Directory where your unzipped burst InSAR products are located')
    parser.add_argument(
        '--filter-strength', type=float, default=0.6, help='Goldstein-Werner filter strength (between 0 and 1)'
    )
    parser.add_argument(
        '--apply-water-mask',
        type=string_is_true,
        default=False,
        help='Apply a water body mask to wrapped and unwrapped phase GeoTIFFs (after unwrapping)',
    )
    args = parser.parse_args()
    product_directory = Path(args.directory)

    prepare_products(product_directory)
    range_looks, azimuth_looks = get_product_multilook(product_directory)
    run_isce2_workflow(
        range_looks, azimuth_looks, filter_strength=args.filter_strength, apply_water_mask=args.apply_water_mask
    )
    package_output(product_directory, f'{range_looks}x{azimuth_looks}', args.filter_strength)


if __name__ == '__main__':
    main()
