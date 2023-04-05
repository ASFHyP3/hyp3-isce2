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
from typing import Union

import requests


S1A_AUX_URL = 'https://sar-mpc.eu/download/55282da1-679d-4ecf-aeef-d06b024451cf'
S1B_AUX_URL = 'https://sar-mpc.eu/download/3c8b7c8d-d3de-4381-a19d-7611fb8734b9'


def _download_platform(url: str, aux_cal_dir: Path):
    """Download and extract the aux cal files for a given satellite platform.

    Args:
        url: URL to download the aux cal files from.
        aux_cal_dir: Directory to download the aux cal files to.
    """
    resp = requests.get(url)
    resp.raise_for_status()

    content = BytesIO(resp.content)
    with zipfile.ZipFile(content) as zip_file:
        zip_file.extractall(aux_cal_dir)


def download_aux_cal(aux_cal_dir: Union[str, Path] = 'aux_cal'):
    """Download and extract the aux cal files for Sentinel-1A/B.

    Args:
        aux_cal_dir: Directory to download the aux cal files to.
    """
    if not isinstance(aux_cal_dir, Path):
        aux_cal_dir = Path(aux_cal_dir)

    aux_cal_dir.mkdir(exist_ok=True, parents=True)
    for url in [S1A_AUX_URL, S1B_AUX_URL]:
        _download_platform(url, aux_cal_dir)


if __name__ == '__main__':
    """Provides a command line interface to download the aux cal files."""
    download_aux_cal()
