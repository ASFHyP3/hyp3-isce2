import os
import shutil
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
        'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85', 'IW1', 'VV', 1
    )
    burst_metadata = burst_utils.BurstMetadata(sample_xml, param)
    ET.ElementTree(burst_metadata.annotation).write(annotation_dir / burst_metadata.annotation_name, **et_args)
    ET.ElementTree(burst_metadata.manifest).write(manifest_dir / f'{burst_metadata.safe_name}.xml', **et_args)
    return annotation_dir, manifest_dir
