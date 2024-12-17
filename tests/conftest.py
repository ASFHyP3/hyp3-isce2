import os
import shutil
from datetime import datetime
from pathlib import Path

import lxml.etree as ET
import pytest

import hyp3_isce2.burst as burst_utils
import hyp3_isce2.merge_tops_bursts as merge


@pytest.fixture()
def test_data_dir():
    here = Path(os.path.dirname(__file__))
    return here / 'data'


@pytest.fixture()
def test_merge_dir(test_data_dir):
    merge_dir = test_data_dir / 'merge'

    if not merge_dir.exists():
        print('Unzipping merge test data...')
        merge_zip = merge_dir.with_suffix('.zip')

        if not merge_zip.exists():
            raise ValueError('merge data not present, run data/create_merge_test_data.py')

        shutil.unpack_archive(merge_zip, test_data_dir)

    return merge_dir


@pytest.fixture()
def annotation_manifest_dirs(tmp_path, test_data_dir):
    annotation_dir, manifest_dir = merge.prep_metadata_dirs(tmp_path)
    sample_xml = ET.parse(test_data_dir / 'reference_descending.xml').getroot()

    et_args = {'encoding': 'UTF-8', 'xml_declaration': True}
    param = burst_utils.BurstParams(
        'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
        'IW2',
        'VV',
        1,
    )
    burst_metadata = burst_utils.BurstMetadata(sample_xml, param)
    ET.ElementTree(burst_metadata.annotation).write(annotation_dir / burst_metadata.annotation_name, **et_args)
    ET.ElementTree(burst_metadata.manifest).write(manifest_dir / f'{burst_metadata.safe_name}.xml', **et_args)
    return annotation_dir, manifest_dir


@pytest.fixture
def burst_product1(test_merge_dir):
    product_path1 = list(test_merge_dir.glob('S1_136231*'))[0]
    product1 = merge.BurstProduct(
        granule='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
        reference_date=datetime(2020, 6, 4, 2, 23, 12),
        secondary_date=datetime(2020, 6, 16, 2, 23, 13),
        burst_id=136231,
        swath='IW2',
        polarization='VV',
        burst_number=7,
        product_path=product_path1,
        n_lines=377,
        n_samples=1272,
        range_looks=20,
        azimuth_looks=4,
        first_valid_line=8,
        n_valid_lines=363,
        first_valid_sample=9,
        n_valid_samples=1220,
        az_time_interval=0.008222225199999992,
        rg_pixel_size=46.59124229430646,
        start_utc=datetime(2020, 6, 4, 2, 23, 13, 963847),
        stop_utc=datetime(2020, 6, 4, 2, 23, 16, 30988),
        relative_orbit=64,
    )
    return product1


@pytest.fixture
def burst_product2(test_merge_dir):
    product_path2 = list(test_merge_dir.glob('*'))[0]

    product2 = merge.BurstProduct(
        granule='S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
        reference_date=datetime(2020, 6, 4, 2, 23, 15),
        secondary_date=datetime(2020, 6, 16, 2, 23, 16),
        burst_id=136232,
        swath='IW2',
        polarization='VV',
        burst_number=8,
        product_path=product_path2,
        n_lines=377,
        n_samples=1272,
        range_looks=20,
        azimuth_looks=4,
        first_valid_line=8,
        n_valid_lines=363,
        first_valid_sample=9,
        n_valid_samples=1220,
        az_time_interval=0.008222225199999992,
        rg_pixel_size=46.59124229430646,
        start_utc=datetime(2020, 6, 4, 2, 23, 16, 722124),
        stop_utc=datetime(2020, 6, 4, 2, 23, 18, 795712),
        relative_orbit=64,
    )
    return product2


@pytest.fixture
def burst_products(burst_product1, burst_product2):
    return [burst_product1, burst_product2]


@pytest.fixture
def test_s1_obj(annotation_manifest_dirs, burst_products):
    annotation_dir, manifest_dir = annotation_manifest_dirs
    s1_obj = merge.load_isce_s1_obj(2, 'VV', annotation_dir.parent)
    s1_obj.output = str(annotation_dir.parent / burst_products[0].swath)
    s1_obj.select_bursts([x.start_utc for x in burst_products])
    s1_obj.update_burst_properties(burst_products)
    return s1_obj


@pytest.fixture
def isce2_merge_setup(annotation_manifest_dirs, burst_products):
    base_dir = annotation_manifest_dirs[0].parent
    s1_obj = merge.create_burst_cropped_s1_obj(2, burst_products, 'VV', base_dir=base_dir)
    for product, burst_obj in zip(burst_products, s1_obj.product.bursts):
        product.isce2_burst_number = burst_obj.burstNumber

    save_dir = str(base_dir / 'fine_interferogram')
    multilooked_swath_obj = merge.modify_for_multilook(burst_products, s1_obj, save_dir)
    multilooked_swath_obj.write_xml()

    merge.spoof_isce2_setup(burst_products, base_dir=base_dir)
    return base_dir
