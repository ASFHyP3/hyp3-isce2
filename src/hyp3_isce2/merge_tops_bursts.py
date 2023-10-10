import argparse
import copy
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from secrets import token_hex
from shutil import make_archive
from tempfile import TemporaryDirectory
from typing import Iterable

import asf_search
import geopandas
import lxml.etree as ET
import numpy as np
import s3fs
import shapely
from hyp3lib.util import string_is_true
from osgeo import gdal
from shapely import geometry
from shapely.geometry import box

import isce  # noqa
import isceobj
from contrib.Snaphu.Snaphu import Snaphu
from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
from isceobj.TopsProc.runMergeBursts import mergeBox, mergeBursts2, multilook
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
from hyp3_isce2.utils import make_browse_image, GDALConfigManager
from hyp3_isce2.water_mask import split_geometry_on_antimeridian

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO, force=True)
log = logging.getLogger(__name__)

from hyp3_isce2.insar_tops_burst import get_pixel_size, translate_outputs  # noqa

os.environ['PATH'] = f"{os.environ.get('PATH')}:{os.environ.get('ISCE_HOME') + '/applications'}"

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
    granule: str
    swath: str
    polarization: str
    burst_number: int
    product_path: Path
    first_valid_line: int
    n_valid_lines: int
    first_valid_sample: int
    n_valid_samples: int
    start_utc: datetime.datetime
    isce2_burst_number: int = field(default=None)

    def to_burst_params(self):
        return burst_utils.BurstParams(self.granule, self.swath, self.polarization, self.burst_number)


def read_product_metadata(meta_file_path):
    hyp3_meta = {}
    with open(meta_file_path) as f:
        for line in f:
            key, value = line.strip().replace(' ', '').split(':')[:2]
            hyp3_meta[key] = value
    return hyp3_meta


def get_burst_metadata(product_paths: Iterable[Path]):
    meta_file_paths = [path / f'{path.name}.txt' for path in product_paths]
    metas = [read_product_metadata(path) for path in meta_file_paths]

    # TODO why does asf_search not return values in order?
    results = [asf_search.granule_search(item['ReferenceGranule'])[0] for item in metas]

    relative_orbits = list(set([result.properties['pathNumber'] for result in results]))
    if len(relative_orbits) > 1:
        msg = (
            'Only burst from the same relative orbit can be merged.'
            f'Currently have bursts from orbits: {", ".join(relative_orbits)}'
        )
        raise ValueError(msg)

    granules = [Path(result.properties['url']).parts[2] for result in results]
    swaths = [result.properties['burst']['subswath'] for result in results]
    burst_indexes = [result.properties['burst']['burstIndex'] for result in results]
    polarization = [result.properties['polarization'] for result in results]
    start_utc = [
        datetime.datetime.strptime(result.properties['startTime'], '%Y-%m-%dT%H:%M:%S.%fZ') for result in results
    ]
    first_valid_line = [meta['Fullresolutionfirstvalidline'] for meta in metas]
    n_valid_lines = [meta['Fullresolutionnumberoflines'] for meta in metas]
    first_valid_sample = [meta['Fullresolutionfirstvalidsample'] for meta in metas]
    n_valid_samples = [meta['Fullresolutionnumberofsamples'] for meta in metas]
    products = []
    for i in range(len(granules)):
        product = BurstProduct(
            granules[i],
            swaths[i],
            polarization[i],
            burst_indexes[i],
            product_paths[i],
            first_valid_line[i],
            n_valid_lines[i],
            first_valid_sample[i],
            n_valid_samples[i],
            start_utc[i],
        )
        products.append(product)
    return products


def download_annotation_xmls(params):
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


def get_scene_roi(s1_obj_bursts):
    for i, isce_burst in enumerate(s1_obj_bursts):
        snwe = isce_burst.getBbox()
        bbox = geometry.box(snwe[2], snwe[0], snwe[3], snwe[1])
        if i == 0:
            overall_bbox = bbox
        else:
            overall_bbox = overall_bbox.union(bbox)
    return overall_bbox.bounds


class Sentinel1BurstSelect(Sentinel1):
    def select_bursts(self, start_utcs):
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

    def update_burst_properties(self, products):
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

    def write_xml(self):
        pm = ProductManager()
        pm.configure()

        outxml = self.output
        if outxml.endswith('/'):
            outxml = outxml[:-1]
        pm.dumpProduct(self.product, os.path.join(outxml + '.xml'))


def create_s1_instance(swath, products, polarization='VV', outdir='fine_interferogram'):
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


def download_dem(s1_objs):
    burst_objs = []
    for s1_obj in s1_objs:
        burst_objs += s1_obj.product.bursts
    dem_roi = get_scene_roi(burst_objs)
    download_dem_for_isce2(dem_roi, dem_name='glo_30', dem_dir=Path.cwd(), buffer=0, resample_20m=False)


def translate_image(in_path, out_path, width, image_type):
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

    gdal.Translate(out_path, in_path, bandList=[n + 1 for n in range(n_bands)], format='ISCE')
    out_img.renderHdr()


def spoof_isce2_setup(burst_products, s1_obj):
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
        width = s1_obj.product.bursts[product.isce2_burst_number - 1].numberOfSamples
        for image_type in file_types:
            if image_type in ['int', 'cor']:
                dir = ifg_dir
                name = f'burst_{product.isce2_burst_number:02}.{image_type}'
            else:
                dir = geom_dir
                name = f'{image_type}_{product.isce2_burst_number:02}.rdr'
            in_path = str(product.product_path / f'{product.product_path.stem}_{file_types[image_type]}.tif')
            out_path = str(dir / product.swath / name)
            translate_image(in_path, out_path, width, image_type)


def getSwathList(indir):
    swathList = []
    for x in [1, 2, 3]:
        swath_paths = os.path.join(indir, 'IW{0}'.format(x))
        if os.path.exists(swath_paths):
            swathList.append(x)

    return swathList


def loadProduct(xmlname):
    '''
    Load the product using Product Manager.
    '''
    from iscesys.Component.ProductManager import ProductManager as PM

    pm = PM()
    pm.configure()
    obj = pm.loadProduct(xmlname)
    return obj


def getMergedOrbit(product):
    ###Create merged orbit
    orb = Orbit()
    orb.configure()

    burst = product[0].bursts[0]
    # Add first burst orbit to begin with
    for sv in burst.orbit:
        orb.addStateVector(sv)

    for pp in product:
        ##Add all state vectors
        for bb in pp.bursts:
            for sv in bb.orbit:
                if (sv.time < orb.minTime) or (sv.time > orb.maxTime):
                    orb.addStateVector(sv)

            bb.orbit = orb

    return orb


def open_image(in_path, image_subtype=None):
    if image_subtype == 'ifg':
        image = isceobj.createIntImage()
    else:
        image = isceobj.createImage()
    image.load(in_path + '.xml')
    image.setAccessMode('read')
    image.createImage()
    return image


def create_image(out_path, width, access_mode, image_subtype='default', finalize=False):
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


def merge_bursts(azimuth_looks, range_looks, mergedir='merged'):
    '''
    Merge burst products to make it look like stripmap.
    Currently will merge interferogram, lat, lon, z and los.
    '''
    frames = []
    burstIndex = []
    swathList = getSwathList(BURST_IFG_DIR)
    for swath in swathList:
        ifg = loadProduct(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath)))
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
        merged_path = out_path + '.full'
        mergeBursts2(frames, burst_paths, burstIndex, box, merged_path, virtual=True, validOnly=True)
        multilook(merged_path, outname=out_path, alks=azimuth_looks, rlks=range_looks)


def goldstein_werner_filter(filter_strength=0.5, mergedir='merged'):
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


def create_water_mask():
    input_image = 'full_res.dem.wgs84'
    output_image = str(Path('merged') / 'water_mask')

    image = isceobj.createDemImage()
    image.load(f'{input_image}.xml')
    image.setAccessMode('read')

    y_size = image.length
    x_size = image.width
    upper_left_x = image.coord1.coordStart
    delta_x = image.coord1.coordDelta
    upper_left_y = image.coord2.coordStart
    delta_y = image.coord2.coordDelta
    geotransform = (upper_left_x, delta_x, 0, upper_left_y, 0, delta_y)
    dst_ds = gdal.GetDriverByName('ISCE').Create(output_image, x_size, y_size)
    dst_ds.SetGeoTransform(geotransform)

    srs = gdal.osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.SetMetadataItem('AREA_OR_POINT', 'Area')

    north = upper_left_y
    west = upper_left_x
    south = north + delta_y * (y_size - 1)
    east = west + delta_x * (x_size - 1)
    extent = box(west, south, east, north)
    extent = split_geometry_on_antimeridian(json.loads(shapely.to_geojson(extent)))

    s3_fs = s3fs.S3FileSystem(anon=True, default_block_size=5 * (2**20))
    with s3_fs.open('asf-dem-west/WATER_MASK/GSHHG/hyp3_water_mask_20220912.parquet', 'rb') as s3_file:
        full_gdf = geopandas.read_parquet(s3_file)
        mask = geopandas.clip(full_gdf, geometry.shape(extent))

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / 'mask.shp'
        mask.to_file(temp_file)
        with GDALConfigManager(OGR_ENABLE_PARTIAL_REPROJECTION='YES'):
            gdal.Rasterize(dst_ds, str(temp_file), allTouched=True, burnValues=[1])

    del dst_ds


def resample_to_radar(maskin, latin, lonin, output):
    maskim = isceobj.createImage()
    maskim.load(maskin + '.xml')
    latim = isceobj.createImage()
    latim.load(latin + '.xml')
    lonim = isceobj.createImage()
    lonim.load(lonin + '.xml')
    mask = np.fromfile(maskin, maskim.toNumpyDataType())
    lat = np.fromfile(latin, latim.toNumpyDataType())
    lon = np.fromfile(lonin, lonim.toNumpyDataType())
    mask = np.reshape(mask, [maskim.coord2.coordSize, maskim.coord1.coordSize])
    startLat = maskim.coord2.coordStart
    deltaLat = maskim.coord2.coordDelta
    startLon = maskim.coord1.coordStart
    deltaLon = maskim.coord1.coordDelta
    lati = np.clip(((lat - startLat) / deltaLat).astype(int), 0, mask.shape[0] - 1)
    loni = np.clip(((lon - startLon) / deltaLon).astype(int), 0, mask.shape[1] - 1)
    cropped = (mask[lati, loni]).astype(maskim.toNumpyDataType())
    cropped = np.reshape(cropped, (latim.coord2.coordSize, latim.coord1.coordSize))
    cropped.tofile(output)
    croppedim = isceobj.createImage()
    croppedim.initImage(output, 'read', cropped.shape[1], maskim.dataType)
    croppedim.renderHdr()


def mask_coherence():
    create_water_mask()
    mask_geo, lat, lon, mask_rdr, coh = ('water_mask', LAT_NAME, LON_NAME, 'water_mask.rdr', COH_NAME)
    mask_geo, lat, lon, mask_rdr, coh = [str(Path('merged') / name) for name in (mask_geo, lat, lon, mask_rdr, coh)]
    resample_to_radar(mask_geo, lat, lon, mask_rdr)
    cmd = f"ImageMath.py -e 'a*b' --a={coh} --b={mask_rdr} -o merged/masked.{COH_NAME}"
    status = os.system(cmd)
    if status != 0:
        raise Exception('error when running:\n{}\n'.format(cmd))


def snaphu_unwrap(
    azimuth_looks, range_looks, corrfile=None, mergedir='merged', cost_mode='DEFO', init_method='MST', defomax=4.0
):
    if not corrfile:
        corrfile = os.path.join(mergedir, COH_NAME)
    wrap_name = os.path.join(mergedir, FILT_WRP_IFG_NAME)
    unwrap_name = os.path.join(mergedir, UNW_IFG_NAME)

    img = isceobj.createImage()
    img.load(wrap_name + '.xml')
    width = img.getWidth()

    swath = getSwathList(BURST_IFG_DIR)[0]
    ifg = loadProduct(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath)))
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


def geocode_products(azimuth_looks, range_looks, dem_filename, mergedir='merged', to_be_geocoded=GEOCODE_LIST):
    to_be_geocoded = [str(Path(mergedir) / file) for file in to_be_geocoded]
    swath_list = getSwathList(BURST_IFG_DIR)

    frames = []
    for swath in swath_list:
        reference_product = loadProduct(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath)))
        frames.append(reference_product)

    orbit = getMergedOrbit(frames)

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
    demImage.load(dem_filename + '.xml')

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
    example_name_split = list(product_directory.glob('S1_??????_IW?_*'))[0].name.split('_')

    swath_number = getSwathList(BURST_IFG_DIR)[0]
    insar_product = loadProduct(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath_number)))

    platform = example_name_split[0]
    relative_orbit = f'{insar_product.bursts[0].trackNumber:03}'
    placeholder = ''
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
            placeholder,
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
    azimuth_looks: int,
    filter_strength: float,
    range_looks: int,
    dem_name: str = 'GLO_30',
    dem_resolution: int = 30,
) -> None:
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

    swath_number = getSwathList(BURST_IFG_DIR)[0]
    insar_product = loadProduct(os.path.join(BURST_IFG_DIR, 'IW{0}.xml'.format(swath_number)))

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


def prepare_products(directory):
    product_paths = list(directory.glob('S1_??????_IW?_*'))
    products = get_burst_metadata(product_paths)
    download_annotation_xmls([product.to_burst_params() for product in products])
    swaths = list(set([int(product.swath[2:3]) for product in products]))
    s1_objs = []
    for swath in swaths:
        swath_products = [product for product in products if int(product.swath[2:3]) == swath]
        swath_products, s1_obj = create_s1_instance(swath, swath_products)
        spoof_isce2_setup(swath_products, s1_obj)
        s1_objs.append(copy.deepcopy(s1_obj))
        del s1_obj

    download_dem(s1_objs)


def run_isce2_workflow(azimuth_looks, range_looks, mergedir='merged', filter_strength=0.5, apply_water_mask=False):
    Path(mergedir).mkdir(exist_ok=True)
    merge_bursts(azimuth_looks, range_looks, mergedir=mergedir)
    goldstein_werner_filter(filter_strength=filter_strength, mergedir=mergedir)
    if apply_water_mask:
        log.info('Water masking requested, downloading water mask')
        mask_coherence()
        corrfile = os.path.join(mergedir, f'masked.{COH_NAME}')
    else:
        corrfile = os.path.join(mergedir, COH_NAME)
    snaphu_unwrap(azimuth_looks, range_looks, corrfile=corrfile, mergedir=mergedir)
    geocode_products(azimuth_looks, range_looks, dem_filename='full_res.dem.wgs84', mergedir=mergedir)


def package_output(product_directory, looks, filter_strength, archive=False):
    pixel_size = get_pixel_size(looks)
    range_looks, azimuth_looks = [int(look) for look in looks.split('x')]

    product_name = get_product_name(product_directory, pixel_size)
    product_dir = Path(product_name)
    product_dir.mkdir(parents=True, exist_ok=True)

    make_parameter_file(
        Path(f'{product_name}/{product_name}.txt'), product_directory, azimuth_looks, range_looks, filter_strength
    )
    translate_outputs(product_name, pixel_size=pixel_size, include_radar=False)
    unwrapped_phase = f'{product_name}/{product_name}_unw_phase.tif'
    make_browse_image(unwrapped_phase, f'{product_name}/{product_name}_unw_phase.png')
    if archive:
        make_archive(base_name=product_name, format='zip', base_dir=product_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', type=str, help='Directory where your unzipped burst InSAR products are located')
    parser.add_argument(
        '--looks', choices=['20x4', '10x2', '5x1'], default='20x4', help='Number of looks to take in range and azimuth'
    )
    parser.add_argument(
        '--filter-strength', type=float, default=0.5, help='Goldstein-Werner filter strength (between 0 and 1)'
    )
    parser.add_argument(
        '--apply-water-mask',
        type=string_is_true,
        default=False,
        help='Apply a water body mask to wrapped and unwrapped phase GeoTIFFs (after unwrapping)',
    )
    args = parser.parse_args()
    range_looks, azimuth_looks = [int(looks) for looks in args.looks.split('x')]
    product_directory = Path(args.directory)

    prepare_products(product_directory)
    run_isce2_workflow(
        azimuth_looks, range_looks, filter_strength=args.filter_strength, apply_water_mask=args.apply_water_mask
    )
    package_output(product_directory, args.looks, args.filter_strength)


if __name__ == '__main__':
    main()
