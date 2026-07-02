# Copyright 2021-present Caltech
# Modifications Copyright 2023 Alaska Satellite Facility
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zipfile
from io import BytesIO
from pathlib import Path

import requests


S1A_AUX_URL = 'https://d3g9emy65n853h.cloudfront.net/AUX_CAL/S1A_AUX_CAL_20241128.zip'
S1B_AUX_URL = 'https://d3g9emy65n853h.cloudfront.net/AUX_CAL/S1B_AUX_CAL_20241128.zip'


def _download_platform(url: str, aux_cal_dir: Path):
    """Download and extract the aux cal files for a given satellite platform.

    Args:
        url: URL to download the aux cal files from.
        aux_cal_dir: Directory to download the aux cal files to.
    """
    response = requests.get(url)
    response.raise_for_status()

    content = BytesIO(response.content)
    with zipfile.ZipFile(content) as zip_file:
        for zip_info in zip_file.infolist():
            # remove leading directories, i.e. extract S1A/AUX_CAL/2019/02/28/foo.SAFE/* to foo.SAFE/*
            if not zip_info.is_dir() and '.SAFE/' in zip_info.filename:
                zip_info.filename = '/'.join(zip_info.filename.split('/')[5:])
                zip_file.extract(zip_info, aux_cal_dir)


def download_aux_cal(aux_cal_dir: str | Path = 'aux_cal'):
    """Download and extract the aux cal files for Sentinel-1A/B.

    Args:
        aux_cal_dir: Directory to download the aux cal files to.
    """
    if not isinstance(aux_cal_dir, Path):
        aux_cal_dir = Path(aux_cal_dir)

    aux_cal_dir.mkdir(exist_ok=True, parents=True)
    for url in (S1A_AUX_URL, S1B_AUX_URL):
        _download_platform(url, aux_cal_dir)


if __name__ == '__main__':
    download_aux_cal()
