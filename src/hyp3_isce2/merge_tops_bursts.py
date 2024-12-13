"""A workflow for merging standard burst InSAR products."""

import argparse
import copy
import datetime
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from secrets import token_hex
from shutil import make_archive
from tempfile import TemporaryDirectory
from typing import Iterable, List, Optional, Tuple

import asf_search
import isce
import isceobj
import lxml.etree as ET
import numpy as np
from contrib.Snaphu.Snaphu import Snaphu
from hyp3lib.util import string_is_true
from isceobj.Orbit.Orbit import Orbit
from isceobj.Planet.Planet import Planet
from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
from isceobj.TopsProc.runIon import maskUnwrap
from isceobj.TopsProc.runMergeBursts import mergeBox, mergeBursts2
from iscesys.Component import createTraitSeq
from iscesys.Component.ProductManager import ProductManager
from mroipac.filter.Filter import Filter
from mroipac.icu.Icu import Icu
from osgeo import gdal
from shapely import geometry
from stdproc.rectify.geocode.Geocodable import Geocodable
from zerodop.geozero import createGeozero

import hyp3_isce2
import hyp3_isce2.burst as burst_utils
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.packaging import get_pixel_size, translate_outputs
from hyp3_isce2.utils import (
    ParameterFile,
    create_image,
    get_projection,
    image_math,
    load_product,
    make_browse_image,
    read_product_metadata,
    resample_to_radar_io,
)
from hyp3_isce2.water_mask import create_water_mask


log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    stream=sys.stdout, format=log_format, level=logging.INFO, force=True
)
log = logging.getLogger(__name__)


BURST_IFG_DIR = "fine_interferogram"
BURST_GEOM_DIR = "geom_reference"
FILT_WRP_IFG_NAME = "filt_topophase.flat"
UNW_IFG_NAME = "filt_topophase.unw"
WRP_IFG_NAME = "topophase.flat"
COH_NAME = "phsig.cor"
LOS_NAME = "los.rdr"
LAT_NAME = "lat.rdr"
LON_NAME = "lon.rdr"
CCOM_NAME = "filt_topophase.unw.conncomp"
GEOCODE_LIST = [UNW_IFG_NAME, COH_NAME, LOS_NAME, FILT_WRP_IFG_NAME, CCOM_NAME]


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
    isce2_burst_number: int | None = None

    def to_burst_params(self):
        """Convert to a burst_utils.BurstParams object"""
        return burst_utils.BurstParams(
            self.granule, self.swath, self.polarization, self.burst_number
        )


def get_burst_metadata(product_paths: list[Path]) -> list[BurstProduct]:
    """Create a list of BurstProduct objects from a set of burst product paths

    Args:
        product_paths: A list of paths to unzipped burst product directories

    Returns:
        A list of BurstProduct objects representing the burst metadata
    """
    meta_file_paths = [path / f"{path.name}.txt" for path in product_paths]
    metas = [read_product_metadata(str(path)) for path in meta_file_paths]

    # TODO why does asf_search not return values in order?
    results = [asf_search.granule_search(item["ReferenceGranule"])[0] for item in metas]

    relative_orbits = [result.properties["pathNumber"] for result in results]
    granules = [Path(result.properties["url"]).parts[2] for result in results]
    pattern = "%Y%m%dT%H%M%S"
    reference_granules = [
        datetime.datetime.strptime(item["ReferenceGranule"].split("_")[3], pattern)
        for item in metas
    ]
    secondary_granules = [
        datetime.datetime.strptime(item["SecondaryGranule"].split("_")[3], pattern)
        for item in metas
    ]
    swaths = [result.properties["burst"]["subswath"] for result in results]
    burst_ids = [result.properties["burst"]["relativeBurstID"] for result in results]
    burst_indexes = [result.properties["burst"]["burstIndex"] for result in results]
    polarization = [result.properties["polarization"] for result in results]

    start_utc_strs = [
        result.umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        for result in results
    ]
    start_utc = [
        datetime.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%S.%fZ")
        for utc in start_utc_strs
    ]
    relative_orbits = [result.properties["pathNumber"] for result in results]
    n_lines = [int(meta["Radarnlines"]) for meta in metas]
    n_samples = [int(meta["Radarnsamples"]) for meta in metas]
    range_looks = [int(meta["Rangelooks"]) for meta in metas]
    azimuth_looks = [int(meta["Azimuthlooks"]) for meta in metas]
    first_valid_line = [int(meta["Radarfirstvalidline"]) for meta in metas]
    n_valid_lines = [int(meta["Radarnvalidlines"]) for meta in metas]
    first_valid_sample = [int(meta["Radarfirstvalidsample"]) for meta in metas]
    n_valid_samples = [int(meta["Radarnvalidsamples"]) for meta in metas]
    az_time_interval = [float(meta["Multilookazimuthtimeinterval"]) for meta in metas]
    rg_pixel_size = [float(meta["Multilookrangepixelsize"]) for meta in metas]
    stop_utc = [
        datetime.datetime.strptime(meta["Radarsensingstop"], "%Y-%m-%dT%H:%M:%S.%f")
        for meta in metas
    ]
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


def prep_metadata_dirs(base_path: Optional[Path] = None) -> Tuple[Path, Path]:
    """Create the annotation and manifest directories

    Args:
        base_path: The base path to create the directories in. Defaults to the current working directory.

    Returns:
        A tuple of the annotation and manifest directories
    """
    if base_path is None:
        base_path = Path.cwd()
    annotation_dir = base_path / "annotation"
    annotation_dir.mkdir(exist_ok=True)
    manifest_dir = base_path / "manifest"
    manifest_dir.mkdir(exist_ok=True)
    return annotation_dir, manifest_dir


def download_metadata_xmls(
    params: Iterable[burst_utils.BurstParams], base_dir: Optional[Path] = None
) -> None:
    """Download annotation xmls for a set of burst parameters to a directory
    named 'annotation' in the current working directory

    Args:
        params: A list of burst_utils.BurstParams objects
        base_dir: The base directory to download the metadata to. Defaults to the current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    annotation_dir, manifest_dir = prep_metadata_dirs(base_dir)

    granules = list(set([param.granule for param in params]))
    download_params = [
        burst_utils.BurstParams(granule, "IW1", "VV", 0) for granule in granules
    ]

    with burst_utils.get_asf_session() as asf_session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            xml_futures = [
                executor.submit(burst_utils.download_metadata, asf_session, param)
                for param in download_params
            ]
            metadata_xmls = {
                param.granule: future.result()
                for param, future in zip(download_params, xml_futures)
            }

    et_args = {"encoding": "UTF-8", "xml_declaration": True}
    for param in params:
        metadata_xml = metadata_xmls[param.granule]
        burst_metadata = burst_utils.BurstMetadata(metadata_xml, param)
        ET.ElementTree(burst_metadata.annotation).write(
            annotation_dir / burst_metadata.annotation_name, **et_args
        )
        ET.ElementTree(burst_metadata.manifest).write(
            manifest_dir / f"{burst_metadata.safe_name}.xml", **et_args
        )


def get_scene_roi(
    s1_obj_bursts: Iterable[isceobj.Sensor.TOPS.BurstSLC.BurstSLC],
) -> Tuple:
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
        start_utcs = sorted(start_utcs)
        cropList = createTraitSeq("burst")
        tiffList = []
        eapList = []

        print("Number of Bursts before cropping: ", len(self.product.bursts))
        for start_utc in start_utcs:
            start_utc = start_utc.replace(microsecond=0)
            match = [
                x
                for x in enumerate(self.product.bursts)
                if x[1].burstStartUTC.replace(microsecond=0) == start_utc
            ]

            if not len(match) > 0:
                raise ValueError(f"No match for burst at time {start_utc} found.")
            ind, isce2_burst = match[0]

            cropList.append(isce2_burst)
            if len(self._tiffSrc): #type: ignore
                tiffList.append(self._tiffSrc[ind]) #type: ignore
            eapList.append(self._elevationAngleVsTau[ind])

        # Actual cropping
        self.product.bursts = cropList
        self.product.numberOfBursts = len(self.product.bursts)

        self._tiffSrc = tiffList #type: ignore
        self._elevationAngleVsTau = eapList #type: ignore
        print("Number of Bursts after cropping: ", len(self.product.bursts))

    def update_burst_properties(self, products: list[BurstProduct]) -> None:
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
            if product.start_utc.replace(microsecond=0) != burst.burstStartUTC.replace(
                microsecond=0
            ):
                raise ValueError(
                    "Burst product and ISCE2 burst do not match (different start times)."
                )

            burst.firstValidLine = product.first_valid_line
            burst.numValidLines = product.n_valid_lines
            burst.firstValidSample = product.first_valid_sample
            burst.numValidSamples = product.n_valid_samples

            outfile = os.path.join(self.output, "burst_%02d" % (index + 1) + ".slc")
            slcImage = isceobj.createSlcImage()
            slcImage.setByteOrder("l")
            slcImage.setFilename(outfile)
            slcImage.setAccessMode("read")
            slcImage.setWidth(width)
            slcImage.setLength(length)
            slcImage.setXmin(0)
            slcImage.setXmax(width)
            burst.image = slcImage
            burst.numberOfSamples = width
            burst.numberOfLines = length

            print(
                "Updating burst number from {0} to {1}".format(
                    burst.burstNumber, index + 1
                )
            )
            burst.burstNumber = index + 1

    def write_xml(self) -> None:
        """Write the product xml to the directory specified by self.output"""
        pm = ProductManager()
        pm.configure()

        outxml = self.output
        if outxml.endswith("/"):
            outxml = outxml[:-1]
        pm.dumpProduct(self.product, os.path.join(outxml + ".xml"))


def load_isce_s1_obj(
    swath: int, polarization: str, base_dir: Optional[Path] = None
) -> Sentinel1BurstSelect:
    """Load a modified ISCE2 Sentinel1 instance for a swath and polarization given annotation and manifest directories

    Args:
        swath: The swath number
        polarization: The polarization
        base_dir: The base directory containing the annotation and manifest directories. Defaults to the woking dir.

    Returns:
        A modified ISCE2 Sentinel1 instance
    """
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    annotation_dir = base_dir / "annotation"
    manifest_dir = base_dir / "manifest"

    annotation_xmls = [
        str(path)
        for path in annotation_dir.glob(f"s1?-??{swath}-slc-{polarization.lower()}*")
    ]
    if len(annotation_xmls) == 0:
        raise ValueError(
            f"No annotation files for swath {swath} and polarization {polarization} found in annotation directory"
        )
    manifest_xmls = [str(path) for path in manifest_dir.glob("S1*.xml")]

    obj = Sentinel1BurstSelect()
    obj.configure()
    obj.xml = annotation_xmls
    obj.tiff = ["" for _ in range(len(annotation_xmls))]
    obj.manifest = manifest_xmls
    obj.swath = swath
    obj.polarization = polarization.lower()
    obj.parse()
    return obj


def create_burst_cropped_s1_obj(
    swath: int,
    products: Iterable[BurstProduct],
    polarization: str = "VV",
    base_dir: Optional[Path] = None,
) -> Sentinel1BurstSelect:
    """Create an ISCE2 Sentinel1BurstSelect instance for a set of burst products, and write the xml.
    Also updates the BurstProduct objects with the ISCE2 burst number.

    Args:
        swath: The swath id (e.g., IW1) of the burst products
        products: A list of BurstProduct objects to create the ISCE2 Sentinel1 instance for
        polarization: The polarization of the burst products
        base_dir: The base directory to write the xml to. Defaults to the current working directory.

    Returns:
        A tuple of the updated BurstProduct objects and the ISCE2 Sentinel1 instance
    """
    if base_dir is None:
        base_dir = Path.cwd()

    swaths_in_products = list(set([int(product.swath[2:3]) for product in products]))
    if len(swaths_in_products) > 1 or swaths_in_products[0] != swath:
        raise ValueError(f"Products provided are not all in swath {swath}")

    obj = load_isce_s1_obj(swath, polarization, base_dir=base_dir)

    out_path = base_dir / BURST_IFG_DIR
    out_path.mkdir(exist_ok=True)
    obj.output = str(out_path / f"IW{swath}")

    products = sorted(products, key=lambda x: x.start_utc)
    obj.select_bursts([b.start_utc for b in products])
    obj.update_burst_properties(products)
    obj.write_xml()
    return obj


def modify_for_multilook(
    burst_products: Iterable[BurstProduct],
    swath_obj: Sentinel1BurstSelect,
    outdir: str = BURST_IFG_DIR,
) -> Sentinel1BurstSelect:
    """Modify a Sentinel1 instance so that it is compatible with previously multilooked burst products

    Args:
        burst_products: A list of BurstProduct objects containing the needed metadata
        swath_obj: A Sentinel1BurstSelect (or Sentinel1) instance representing the parent swath
        outdir: The directory to write the xml to
    """
    multilook_swath_obj = copy.deepcopy(swath_obj)
    multilook_swath_obj.output = os.path.join(
        outdir, "IW{0}_multilooked".format(multilook_swath_obj.swath)
    )
    multilook_swath_obj.output = str(
        Path(outdir) / f"IW{multilook_swath_obj.swath}_multilooked"
    )
    for new_metadata, burst_obj in zip(
        burst_products, multilook_swath_obj.product.bursts
    ):
        if new_metadata.start_utc.replace(
            microsecond=0
        ) != burst_obj.burstStartUTC.replace(microsecond=0):
            raise ValueError(
                "Burst product and ISCE2 burst do not match (different start times)."
            )
        burst_obj.numberOfSamples = new_metadata.n_samples
        burst_obj.numberOfLines = new_metadata.n_lines
        burst_obj.firstValidSample = new_metadata.first_valid_sample
        burst_obj.numValidSamples = new_metadata.n_valid_samples
        burst_obj.firstValidLine = new_metadata.first_valid_line
        burst_obj.numValidLines = new_metadata.n_valid_lines
        burst_obj.sensingStop = new_metadata.stop_utc
        burst_obj.azimuthTimeInterval = new_metadata.az_time_interval
        burst_obj.rangePixelSize = new_metadata.rg_pixel_size

    return multilook_swath_obj


def download_dem_for_multiple_bursts(
    s1_objs: Iterable[Sentinel1BurstSelect], base_dir=None
):
    """Download the DEM for the region covered in a set of ISCE2 Sentinel1 instances

    Args:
        s1_objs: A list of Sentinel1BurstSelect instances
        base_dir: The base directory to download the DEM to. Defaults to the current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    burst_objs = []
    for s1_obj in s1_objs:
        burst_objs += s1_obj.product.bursts
    dem_roi = get_scene_roi(burst_objs)
    download_dem_for_isce2(
        dem_roi, dem_name="glo_30", dem_dir=base_dir, buffer=0, resample_20m=False
    )


def translate_image(in_path: str, out_path: str, image_type: str) -> None:
    """Translate a HyP3 burst product image to an ISCE2 compatible image

    Args:
        in_path: The path to the input image
        out_path: The path to the output image
        image_type: The type of image to translate can be one of 'ifg', 'los', 'lat', or 'lon'
    """
    info = gdal.Info(in_path, format="json")
    width = info["size"][0]
    if image_type in "ifg":
        out_img = isceobj.createIntImage()
        n_bands = 1
        out_img.initImage(out_path, "read", width, bands=n_bands)
    elif image_type in ["lat", "lon"]:
        n_bands = 1
        out_img = isceobj.createImage()
        out_img.initImage(out_path, "read", width, "DOUBLE", bands=n_bands)
    elif image_type == "los":
        out_img = isceobj.createImage()
        n_bands = 2
        out_img.initImage(
            out_path, "read", width, "DOUBLE", bands=n_bands, scheme="bil"
        )
        out_img.imageType = "bil"
    else:
        raise NotImplementedError(f"{image_type} is not a valid format")

    with TemporaryDirectory() as tmpdir:
        out_tmp_path = str(Path(tmpdir) / Path(out_path).name)
        gdal.Translate(
            out_tmp_path,
            in_path,
            bandList=[n + 1 for n in range(n_bands)],
            format="ENVI",
            creationOptions=["INTERLEAVE=BIL"],
        )
        shutil.copy(out_tmp_path, out_path)
    out_img.renderHdr()


def spoof_isce2_setup(
    burst_products: Iterable[BurstProduct],
    base_dir: Optional[Path] = None,
) -> None:
    """For a set of ASF burst products, create spoofed geom_reference and fine_interferogram directories
    that are in the state they would be in after running topsApp.py from the 'startup' step to the 'burstifg' step.

    Args:
        burst_products: A list of BurstProduct objects
        base_dir: The base directory to write the spoofed directories to. Defaults to the current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    ifg_dir = base_dir / "fine_interferogram"
    ifg_dir.mkdir(exist_ok=True)

    geom_dir = base_dir / "geom_reference"
    geom_dir.mkdir(exist_ok=True)

    swaths = list(set([product.swath for product in burst_products]))
    for swath in swaths:
        ifg_swath_path = ifg_dir / swath
        ifg_swath_path.mkdir(exist_ok=True)

        geom_swath_path = geom_dir / swath
        geom_swath_path.mkdir(exist_ok=True)

    file_types = {
        "ifg": "wrapped_phase_rdr",
        "los": "los_rdr",
        "lat": "lat_rdr",
        "lon": "lon_rdr",
    }
    for product in burst_products:
        for image_type in file_types:
            if image_type == "ifg":
                img_dir = ifg_dir
                name = f"burst_{product.isce2_burst_number:02}.int"
            else:
                img_dir = geom_dir
                name = f"{image_type}_{product.isce2_burst_number:02}.rdr"
            in_path = str(
                product.product_path
                / f"{product.product_path.stem}_{file_types[image_type]}.tif"
            )
            out_path = str(img_dir / product.swath / name)
            # translate_image(in_path, out_path, product.n_samples, image_type)
            translate_image(in_path, out_path, image_type)


def get_swath_list(base_dir: Path) -> list[str]:
    """Get the list of swaths from a directory of burst products

    Args:
        base_dir: The directory containing the burst products

    Returns:
        A list of swaths
    """
    swathList = []
    for x in [1, 2, 3]:
        swath_path = Path(base_dir) / f"IW{x}"
        if swath_path.exists():
            swathList.append(str(x))

    return swathList


def get_merged_orbit(products: list[Sentinel1]) -> Orbit:
    """Create a merged orbit from a set of ISCE2 Sentinel1 products

    Args:
        product: A list of ISCE2 Sentinel1 products

    Returns:
        The merged orbit
    """
    # Create merged orbit
    orb = Orbit()
    orb.configure()

    burst = products[0].bursts[0]
    # Add first burst orbit to begin with
    for sv in burst.orbit:
        orb.addStateVector(sv)

    for pp in products:
        # Add all state vectors
        for bb in pp.bursts:
            for sv in bb.orbit:
                if (sv.time < orb.minTime) or (sv.time > orb.maxTime):
                    orb.addStateVector(sv)

            bb.orbit = orb

    return orb


def get_frames_and_indexes(burst_ifg_dir: Path) -> Tuple:
    """Get the frames and burst indexes from a directory of burst interferograms.

    Args:
        burst_ifg_dir: The directory containing the burst interferograms

    Returns:
        A tuple of the frames and burst indexes
    """
    frames = []
    burst_index = []

    swath_list = get_swath_list(burst_ifg_dir)
    for swath in swath_list:
        ifg = load_product(
            os.path.join(burst_ifg_dir, "IW{0}_multilooked.xml".format(swath))
        )
        min_burst = ifg.bursts[0].burstNumber - 1
        max_burst = ifg.bursts[-1].burstNumber
        frames.append(ifg)
        burst_index.append([int(swath), min_burst, max_burst])

    return frames, burst_index


def merge_bursts(
    range_looks: int, azimuth_looks: int, merge_dir: str = "merged"
) -> None:
    """Merge burst products into a multi-swath product, and multilook.
    Note: Can't test this function without polluting home directory with ISCE2 files since
            mergeBursts2 sets up pathing assuming it is in the working directory.

    Args:
        azimuth_looks: The number of azimuth looks
        range_looks: The number of range looks
        merge_dir: The directory to write the merged product to
    """
    frames, burstIndex = get_frames_and_indexes(Path(BURST_IFG_DIR))

    box = mergeBox(frames)
    file_types = {
        "ifg": (BURST_IFG_DIR, "burst_%02d.int", WRP_IFG_NAME),
        "los": (BURST_GEOM_DIR, "los_%02d.rdr", LOS_NAME),
        "lat": (BURST_GEOM_DIR, "lat_%02d.rdr", LAT_NAME),
        "lon": (BURST_GEOM_DIR, "lon_%02d.rdr", LON_NAME),
    }

    for file_type in file_types:
        directory, file_pattern, name = file_types[file_type]
        burst_paths = os.path.join(directory, "IW%d", file_pattern)
        out_path = os.path.join(merge_dir, name)
        mergeBursts2(
            frames, burst_paths, burstIndex, box, out_path, virtual=True, validOnly=True
        )
        with TemporaryDirectory() as tmpdir:
            out_tmp_path = str(Path(tmpdir) / Path(out_path).name)
            gdal.Translate(
                out_tmp_path,
                out_path + ".vrt",
                format="ENVI",
                creationOptions=["INTERLEAVE=BIL"],
            )
            shutil.copy(out_tmp_path, out_path)


def goldstein_werner_filter(
    in_path: Path, out_path: Path, coh_path: Path, filter_strength: float = 0.6
) -> None:
    """Apply the Goldstein-Werner filter to the merged interferogram.
    See https://doi.org/10.1029/1998GL900033 for method details.

    Args:
        in_path: The path to the input interferogram
        out_path: The path to the output filtered interferogram
        coh_path: The path to the coherence file
        filter_strength: The filter strength (0,1]
    """
    ifg_image = create_image(str(in_path), image_subtype="ifg", action="load")
    width_ifg = ifg_image.getWidth()

    filt_image = create_image(str(out_path), width_ifg, "write", image_subtype="ifg")

    filter_obj = Filter()
    filter_obj.wireInputPort(name="interferogram", object=ifg_image)
    filter_obj.wireOutputPort(name="filtered interferogram", object=filt_image)
    filter_obj.goldsteinWerner(alpha=filter_strength)

    ifg_image.finalizeImage()
    filt_image.finalizeImage()
    del filt_image

    filt_image = create_image(str(out_path), width_ifg, "read", image_subtype="ifg")
    phsigImage = create_image(str(coh_path), width_ifg, "write", image_subtype="cor")

    icu_obj = Icu(name="topsapp_filter_icu")
    icu_obj.configure()
    icu_obj.unwrappingFlag = False
    icu_obj.useAmplitudeFlag = False
    icu_obj.icu(intImage=filt_image, phsigImage=phsigImage)

    filt_image.finalizeImage()
    phsigImage.finalizeImage()
    phsigImage.renderHdr()


def mask_coherence(out_name: str, merge_dir: Optional[Path] = None) -> None:
    """Mask the coherence image with a water mask that has been resampled to radar coordinates

    Args:
        out_name: The name of the output masked coherence file
        mergedir: The output directory containing the merged interferogram
    """
    if merge_dir is None:
        merge_dir = Path.cwd() / "merged"
    merge_dir = Path(merge_dir)

    input_files = (
        "water_mask",
        LAT_NAME,
        LON_NAME,
        "water_mask.rdr",
        COH_NAME,
        out_name,
    )
    mask_geo, lat, lon, mask_rdr, coh, masked_coh = [
        str(Path(merge_dir) / name) for name in input_files
    ]
    create_water_mask(
        input_image="full_res.dem.wgs84", output_image=mask_geo, gdal_format="ISCE"
    )
    resample_to_radar_io(mask_geo, lat, lon, mask_rdr)
    image_math(coh, mask_rdr, masked_coh, "a*b")


def snaphu_unwrap(
    range_looks: int,
    azimuth_looks: int,
    corrfile: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    cost_mode="DEFO",
    init_method="MST",
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
    if base_dir is None:
        base_dir = Path.cwd() / "merged"
    base_dir = Path(base_dir)

    burst_ifg_dir = base_dir.parent / BURST_IFG_DIR
    if not corrfile:
        corrfile = base_dir / COH_NAME
    wrap_name = base_dir / FILT_WRP_IFG_NAME
    unwrap_name = base_dir / UNW_IFG_NAME

    img = isceobj.createImage()
    img.load(str(wrap_name) + ".xml")
    width = img.getWidth()

    swath = get_swath_list(burst_ifg_dir)[0]
    ifg = load_product(str(burst_ifg_dir / f"IW{swath}_multilooked.xml"))
    wavelength = ifg.bursts[0].radarWavelength

    # tmid
    tstart = ifg.bursts[0].sensingStart
    tend = ifg.bursts[-1].sensingStop
    tmid = tstart + 0.5 * (tend - tstart)

    # Sometimes tmid may exceed the time span, so use mid burst instead
    burst_index = int(np.around(len(ifg.bursts) / 2))
    orbit = ifg.bursts[burst_index].orbit
    peg = orbit.interpolateOrbit(tmid, method="hermite")

    # Calculate parameters for SNAPHU
    ref_elp = Planet(pname="Earth").ellipsoid
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
    snp.setInput(str(wrap_name))
    snp.setOutput(str(unwrap_name))
    snp.setWidth(width)
    snp.setCostMode(cost_mode)
    snp.setEarthRadius(earth_radius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(str(corrfile))
    snp.setInitMethod(init_method)
    snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(maxComponents)
    snp.setDefoMaxCycles(defomax)
    snp.setRangeLooks(range_looks)
    snp.setAzimuthLooks(azimuth_looks)
    snp.setCorFileFormat("FLOAT_DATA")
    snp.prepare()
    snp.unwrap()

    # Render XML
    create_image(
        str(unwrap_name), width, "read", image_subtype="unw", action="finalize"
    )

    # Check if connected components was created
    if not snp.dumpConnectedComponents:
        raise RuntimeError("SNAPHU did not create connected components file")

    create_image(
        str(unwrap_name) + ".conncomp",
        width,
        "read",
        image_subtype="conncomp",
        action="finalize",
    )
    maskUnwrap(str(unwrap_name), str(wrap_name))


def geocode_products(
    range_looks: int,
    azimuth_looks: int,
    dem_path: Path,
    base_dir: Optional[Path] = None,
    to_be_geocoded=GEOCODE_LIST,
) -> None:
    """Geocode a set of ISCE2 products

    Args:
        azimuth_looks: The number of azimuth looks
        range_looks: The number of range looks
        dem_path: The path to the DEM
        base_dir: The output directory containing the merged interferogram
        to_be_geocoded: A list of products to geocode
    """
    if base_dir is None:
        base_dir = Path.cwd() / "merged"
    base_dir = Path(base_dir)
    burst_ifg_dir = base_dir.parent / BURST_IFG_DIR

    to_be_geocoded = [str(base_dir / file) for file in to_be_geocoded]
    swath_list = get_swath_list(burst_ifg_dir)

    frames = []
    for swath in swath_list:
        reference_product = load_product(str(burst_ifg_dir / f"IW{swath}.xml"))
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
    planet = Planet(pname="Earth")

    # Setup DEM
    demImage = isceobj.createDemImage()
    demImage.load(dem_path.with_suffix('.xml'))

    # Geocode one by one
    ge = Geocodable()
    for prod in to_be_geocoded:
        geo_obj = createGeozero()
        geo_obj.configure()

        geo_obj.snwe = snwe
        geo_obj.demCropFilename = os.path.join(base_dir, "dem.crop")
        geo_obj.numberRangeLooks = range_looks
        geo_obj.numberAzimuthLooks = azimuth_looks
        geo_obj.lookSide = -1

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
        geo_obj.setSensingStart(
            t0 + datetime.timedelta(seconds=(((azimuth_looks - 1) / 2.0) * dtaz))
        )
        geo_obj.wireInputPort(name="dem", object=demImage)
        geo_obj.wireInputPort(name="planet", object=planet)
        geo_obj.wireInputPort(name="tobegeocoded", object=inImage)

        geo_obj.geocode()


def get_product_name(product: BurstProduct, pixel_size: float) -> str:
    """Create a product name for a merged burst product. Follows the convention used by ASF burst products,
    but replaces the burst id with the relative orbit number, and removes the swath compenent with ''.

    Args:
        product: The BurstProduct object to create the name for. Can be any product in the merged product.
        pixel_size: The pixel size of the product

    Returns:
        The merged product name
    """
    swath_placeholder = ""
    reference_date = product.reference_date.strftime("%Y%m%d")
    secondary_date = product.secondary_date.strftime("%Y%m%d")
    product_id = token_hex(2).upper()
    return "_".join(
        [
            "S1",
            f"{product.relative_orbit:03d}",
            swath_placeholder,
            reference_date,
            secondary_date,
            product.polarization.upper(),
            f"INT{int(pixel_size)}",
            product_id,
        ]
    )


def get_product_metadata_info(base_dir: Path) -> List:
    """Get the metadata for a set of ASF burst products

    Args:
        base_dir: The directory containing UNZIPPED ASF burst products

    Returns:
        A list of metadata dictionaries
    """
    product_paths = list(Path(base_dir).glob("S1_??????_IW?_*"))
    meta_file_paths = [path / f"{path.name}.txt" for path in product_paths]
    metas = [read_product_metadata(str(path)) for path in meta_file_paths]
    return metas


def make_parameter_file(
    out_path: Path,
    metas: List,
    range_looks: int,
    azimuth_looks: int,
    filter_strength: float,
    water_mask: bool,
    dem_name: str = "GLO_30",
    dem_resolution: int = 30,
    base_dir: Optional[Path] = None,
):
    """Create a parameter file for the ASF merged burst product and write it to out_path

    Args:
        out_path: The path to write the parameter file to
        metas: A list of metadata dictionaries for the burst products
        range_looks: The number of range looks
        azimuth_looks: The number of azimuth looks
        filter_strength: The Goldstein-Werner filter strength
        water_mask: Whether or not to use a water mask
        dem_name: The name of the source DEM
        dem_resolution: The resolution of the source DEM
        base_dir: The base directory to write the parameter file to. Defaults to the current working directory.
    """
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    SPACECRAFT_HEIGHT = 693000.0
    EARTH_RADIUS = 6337286.638938101

    reference_scenes = [meta["ReferenceGranule"] for meta in metas]
    secondary_scenes = [meta["SecondaryGranule"] for meta in metas]
    ref_orbit_number = metas[0]["ReferenceOrbitNumber"]
    sec_orbit_number = metas[0]["SecondaryOrbitNumber"]
    baseline_perp = metas[0]["Baseline"]

    burst_ifg_dir = base_dir / BURST_IFG_DIR
    insar_product = load_product(
        str(burst_ifg_dir / f"IW{get_swath_list(burst_ifg_dir)[0]}.xml")
    )

    orbit_direction = insar_product.bursts[0].passDirection
    ref_heading = insar_product.orbit.getHeading()
    ref_time = insar_product.bursts[0].sensingStart

    utc_time = (
        (((ref_time.hour * 60) + ref_time.minute) * 60)
        + ref_time.second
        + (ref_time.microsecond * 10e-7)
    )
    slant_range_near = insar_product.startingRange
    slant_range_center = insar_product.midRange
    slant_range_far = insar_product.farRange

    parameter_file = ParameterFile(
        reference_granule=", ".join(reference_scenes),
        secondary_granule=", ".join(secondary_scenes),
        reference_orbit_direction=orbit_direction,
        reference_orbit_number=ref_orbit_number,
        secondary_orbit_direction=orbit_direction,
        secondary_orbit_number=sec_orbit_number,
        baseline=baseline_perp,
        utc_time=utc_time,
        heading=ref_heading,
        spacecraft_height=SPACECRAFT_HEIGHT,
        earth_radius_at_nadir=EARTH_RADIUS,
        slant_range_near=slant_range_near,
        slant_range_center=slant_range_center,
        slant_range_far=slant_range_far,
        range_looks=range_looks,
        azimuth_looks=azimuth_looks,
        insar_phase_filter=True,
        phase_filter_parameter=filter_strength,
        range_bandpass_filter=False,
        azimuth_bandpass_filter=False,
        dem_source=dem_name,
        dem_resolution=dem_resolution,
        unwrapping_type="snaphu_mcf",
        speckle_filter=True,
        water_mask=water_mask,
    )
    parameter_file.write(out_path)


def make_readme(
    product_dir: Path,
    reference_scenes: list[str],
    secondary_scenes: list[str],
    range_looks: int,
    azimuth_looks: int,
    apply_water_mask: bool,
) -> None:
    """Create a README file for the merged burst product and write it to product_dir

    Args:
        product_dir: The path to the directory containing the merged burst product,
            the directory name should be the product name
        reference_scenes: A list of reference scenes
        secondary_scenes: A list of secondary scenes
        range_looks: The number of range looks
        azimuth_looks: The number of azimuth looks
        apply_water_mask: Whether or not a water mask was applied
    """
    product_name = product_dir.name
    wrapped_phase_path = product_dir / f"{product_name}_wrapped_phase.tif"
    info = gdal.Info(str(wrapped_phase_path), format="json")
    secondary_granule_datetime_str = secondary_scenes[0].split("_")[3]

    payload = {
        "processing_date": datetime.datetime.now(datetime.timezone.utc),
        "plugin_name": hyp3_isce2.__name__,
        "plugin_version": hyp3_isce2.__version__,
        "processor_name": isce.__name__.upper(),  # noqa
        "processor_version": isce.__version__,  # noqa
        "projection": get_projection(info["coordinateSystem"]["wkt"]),
        "pixel_spacing": info["geoTransform"][1],
        "product_name": product_name,
        "reference_burst_name": ", ".join(reference_scenes),
        "secondary_burst_name": ", ".join(secondary_scenes),
        "range_looks": range_looks,
        "azimuth_looks": azimuth_looks,
        "secondary_granule_date": datetime.datetime.strptime(
            secondary_granule_datetime_str, "%Y%m%dT%H%M%S"
        ),
        "dem_name": "GLO-30",
        "dem_pixel_spacing": "30 m",
        "apply_water_mask": apply_water_mask,
    }
    content = hyp3_isce2.metadata.util.render_template(
        "insar_burst/insar_burst_merge_readme.md.txt.j2", payload
    )

    output_file = product_dir / f"{product_name}_README.md.txt"
    with open(output_file, "w") as f:
        f.write(content)


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
    looks = set([f"{rgl}x{azl}" for rgl, azl in zip(range_looks, azimuth_looks)])

    sets = {
        "reference_date": reference_dates,
        "secondary_date": secondary_dates,
        "polarization": polarizations,
        "relative_orbit": relative_orbits,
        "looks": looks,
    }
    for key, value in sets.items():
        if len(value) > 1:
            key_name = key.replace("_", " ")
            value_names = ", ".join([str(v) for v in value])
            raise ValueError(
                f"All products must have the same {key_name}. Found {value_names}."
            )

    swath_ids = {}
    for swath in set([product.swath for product in products]):
        swath_products = [product for product in products if product.swath == swath]
        swath_products.sort(key=lambda x: x.burst_id)
        ids = np.array([p.burst_id for p in swath_products])
        if not np.all(ids - ids.min() == np.arange(len(ids))):
            raise ValueError(f"Products for swath {swath} are not contiguous")
        swath_ids[swath] = ids

    for swath1, swath2 in combinations(swath_ids.keys(), 2):
        separations = np.concatenate(
            [swath_ids[swath1] - elem for elem in swath_ids[swath2]]
        )
        if separations.min() > 1:
            raise ValueError(
                f"Products from swaths {swath1} and {swath2} do not overlap"
            )


def get_product_multilook(product_dir: Path) -> Tuple:
    """Get the multilook values for a set of ASF burst products.
    You should have already checked that all products have the same multilook,
    so you can just use the first product's values.

    Args:
        product_dir: The path to the directory containing the UNZIPPED ASF burst product directories

    Returns:
        The number of azimuth looks and range looks
    """
    product_path = list(product_dir.glob("S1_??????_IW?_*"))[0]
    metadata_path = str(product_path / f"{product_path.name}.txt")
    meta = read_product_metadata(metadata_path)
    return int(meta["Rangelooks"]), int(meta["Azimuthlooks"])


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
    product_paths = list(directory.glob("S1_??????_IW?_*"))
    products = get_burst_metadata(product_paths)
    check_burst_group_validity(products)
    download_metadata_xmls([product.to_burst_params() for product in products])
    swaths = list(set([int(product.swath[2:3]) for product in products]))
    swath_objs = []
    for swath in swaths:
        swath_products = [
            product for product in products if int(product.swath[2:3]) == swath
        ]
        swath_products = sorted(swath_products, key=lambda x: x.start_utc)
        swath_obj = create_burst_cropped_s1_obj(swath, swath_products)
        for product, burst_obj in zip(swath_products, swath_obj.product.bursts):
            product.isce2_burst_number = burst_obj.burstNumber

        multilooked_swath_obj = modify_for_multilook(swath_products, swath_obj)
        multilooked_swath_obj.write_xml()

        spoof_isce2_setup(swath_products, swath_obj)
        swath_objs.append(copy.deepcopy(swath_obj))
        del swath_obj

    download_dem_for_multiple_bursts(swath_objs)


def run_isce2_workflow(
    range_looks: int,
    azimuth_looks: int,
    mergedir="merged",
    filter_strength=0.5,
    apply_water_mask=False,
) -> None:
    """Run the ISCE2 workflow for burst merging, filtering, unwrapping, and geocoding

    Args:
        azimuth_looks: The number of azimuth looks
        range_looks: The number of range looks
        mergedir: The output directory containing the merged interferogram
        filter_strength: The Goldstein-Werner filter strength
        apply_water_mask: Whether or not to apply a water body mask to the coherence file before unwrapping
    """
    mergedir = Path(mergedir)
    mergedir.mkdir(exist_ok=True)
    merge_bursts(range_looks, azimuth_looks, merge_dir=str(mergedir))
    goldstein_werner_filter(
        mergedir / WRP_IFG_NAME,
        mergedir / FILT_WRP_IFG_NAME,
        mergedir / COH_NAME,
        filter_strength=filter_strength,
    )
    if apply_water_mask:
        log.info("Water masking requested, downloading water mask")
        mask = f"masked.{COH_NAME}"
        mask_coherence(mask)
        corrfile = mergedir / mask
    else:
        corrfile = mergedir / COH_NAME
    snaphu_unwrap(range_looks, azimuth_looks, corrfile=corrfile, base_dir=mergedir)
    geocode_products(
        range_looks,
        azimuth_looks,
        dem_path=Path("full_res.dem.wgs84"),
        base_dir=mergedir,
    )


def package_output(
    product_directory: Path,
    looks: str,
    filter_strength: float,
    water_mask: bool,
    archive=False,
) -> None:
    """Package the output of the ISCE2 workflow into a the standard ASF burst product format

    Args:
        product_directory: The path to the directory containing the UNZIPPED ASF burst product directories
        looks: The number of looks [20x4, 10x2, 5x1]
        filter_strength: The Goldstein-Werner filter strength
        archive: Whether or not to create a zip archive of the output
    """
    pixel_size = get_pixel_size(looks)
    range_looks, azimuth_looks = [int(look) for look in looks.split("x")]

    product_path = Path(list(product_directory.glob("S1_??????_IW?_*"))[0])
    example_metadata = get_burst_metadata([product_path])[0]
    product_name = get_product_name(example_metadata, pixel_size)
    out_product_dir = Path(product_name)
    out_product_dir.mkdir(parents=True, exist_ok=True)

    metas = get_product_metadata_info(product_directory)
    make_parameter_file(
        Path(f"{product_name}/{product_name}.txt"),
        metas,
        range_looks,
        azimuth_looks,
        filter_strength,
        water_mask,
    )

    translate_outputs(product_name, pixel_size=pixel_size, include_radar=False)
    reference_scenes = [meta["ReferenceGranule"] for meta in metas]
    secondary_scenes = [meta["SecondaryGranule"] for meta in metas]
    make_readme(
        out_product_dir,
        reference_scenes,
        secondary_scenes,
        range_looks,
        azimuth_looks,
        water_mask,
    )
    unwrapped_phase = f"{product_name}/{product_name}_unw_phase.tif"
    make_browse_image(unwrapped_phase, f"{product_name}/{product_name}_unw_phase.png")
    if archive:
        make_archive(base_name=product_name, format="zip", base_dir=product_name)


def merge_tops_bursts(
    product_directory: Path, filter_strength: float, apply_water_mask: bool
):
    """Run the full ISCE2 workflow for TOPS burst merging

    Args:
        product_directory: The path to the directory containing the UNZIPPED ASF burst product directories
        filter_strength: The Goldstein-Werner filter strength
        apply_water_mask: Whether or not to apply a water body mask to the coherence file before unwrapping
    """
    range_looks, azimuth_looks = get_product_multilook(product_directory)
    prepare_products(product_directory)
    run_isce2_workflow(
        range_looks,
        azimuth_looks,
        filter_strength=filter_strength,
        apply_water_mask=apply_water_mask,
    )
    package_output(
        product_directory,
        f"{range_looks}x{azimuth_looks}",
        filter_strength,
        water_mask=apply_water_mask,
    )


def main():
    """HyP3 entrypoint for the TOPS burst merging workflow"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory where your unzipped burst InSAR products are located",
    )
    parser.add_argument(
        "--filter-strength",
        type=float,
        default=0.6,
        help="Goldstein-Werner filter strength (between 0 and 1)",
    )
    parser.add_argument(
        "--apply-water-mask",
        type=string_is_true,
        default=False,
        help="Apply a water body mask to wrapped and unwrapped phase GeoTIFFs (after unwrapping)",
    )
    args = parser.parse_args()
    product_directory = Path(args.directory)
    merge_tops_bursts(product_directory, args.filter_strength, args.apply_water_mask)


if __name__ == "__main__":
    main()
