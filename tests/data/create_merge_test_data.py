import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from osgeo import gdal


def create_product(tmp_path: Path, out_path: Path, granule1: str, granule2: str):
    cwd = Path.cwd()

    tmp_path.mkdir(exist_ok=True)
    os.chdir(tmp_path)

    cmd = f'insar_tops_burst {granule1} {granule2}'
    subprocess.run(cmd.split(' '), check=True)
    result_directory = [x for x in tmp_path.glob('S1_*') if x.is_dir()][0].absolute()

    not_radar = [x for x in result_directory.glob('*') if 'rdr' not in x.name]
    also_not_metadata = [x for x in not_radar if not x.name == f'{x.parent.name}.txt']
    for output_file in also_not_metadata:
        output_file.unlink()

    os.chdir(cwd)
    shutil.copytree(result_directory, out_path / result_directory.name)
    shutil.rmtree(result_directory.parent)


def replace_geotiff_data(geotiff_path: str, stock_value: complex) -> None:
    """Load a geotiff file, swap the non-zero data for all ones and resave."""
    ds = gdal.Open(geotiff_path, gdal.GA_Update)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    data[data != 0 + 0j] = stock_value
    band.WriteArray(data)
    ds.FlushCache()
    ds = None


def create_test_products():
    out_dir = Path.cwd() / 'merge'

    pairs = [
        [
            'S1_136231_IW2_20200604T022312_VV_7C85-BURST',
            'S1_136231_IW2_20200616T022313_VV_5D11-BURST',
        ],
        [
            'S1_136232_IW2_20200604T022315_VV_7C85-BURST',
            'S1_136232_IW2_20200616T022316_VV_5D11-BURST',
        ],
    ]
    for granule1, granule2 in pairs:
        create_product(Path.cwd() / 'tmp', out_dir, granule1, granule2)

    ifgs = out_dir.glob('./*/*_wrapped_phase_rdr.tif')
    value = 1 + 0j
    for ifg in ifgs:
        value += (0 + 1j) * np.pi / 2
        replace_geotiff_data(str(ifg), value)

    shutil.make_archive(base_name=out_dir.name, format='zip', base_dir=out_dir.name)
    shutil.rmtree(out_dir)


if __name__ == '__main__':
    create_test_products()
