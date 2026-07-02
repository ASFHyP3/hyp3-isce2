"""Create a full SLC Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow"""

import logging
from pathlib import Path
from shutil import copyfile

from isceobj.TopsProc.runMergeBursts import multilook  # type: ignore[import-not-found]
from s1_orbits import fetch_for_scene

from hyp3_isce2 import slc, topsapp
from hyp3_isce2.dem import download_dem_for_isce2
from hyp3_isce2.s1_auxcal import download_aux_cal
from hyp3_isce2.utils import (
    image_math,
    isce2_copy,
    resample_to_radar_io,
)
from hyp3_isce2.water_mask import create_water_mask


log = logging.getLogger(__name__)


def insar_tops(
    reference_safe_dir: Path,
    secondary_safe_dir: Path,
    swaths: list = [1, 2, 3],
    polarization: str = 'VV',
    azimuth_looks: int = 4,
    range_looks: int = 20,
    apply_water_mask: bool = False,
) -> Path:
    """Create a full-SLC interferogram

    Args:
        reference_safe_dir: Reference SLC SAFE directory
        secondary_safe_dir: Secondary SLC SAFE directory
        swaths: Swaths to process
        polarization: Polarization to use
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks
        apply_water_mask: Apply water mask to unwrapped phase

    Returns:
        Path to the output files
    """
    orbit_dir = Path('orbits')
    aux_cal_dir = Path('aux_cal')
    dem_dir = Path('dem')

    roi = slc.get_dem_bounds(reference_safe_dir, secondary_safe_dir)
    log.info(f'DEM ROI: {roi}')

    dem_dir.mkdir(exist_ok=True, parents=True)
    dem_path = dem_dir / 'full_res.dem.wgs84'
    download_dem_for_isce2(roi, dem_path=dem_path, pixel_size=30.0)
    geocode_dem_path = dem_path
    if range_looks == 5:
        geocode_dem_path = dem_dir / 'full_res_geocode.dem.wgs84'
        download_dem_for_isce2(roi, dem_path=geocode_dem_path, pixel_size=20.0)

    download_aux_cal(aux_cal_dir)

    orbit_dir.mkdir(exist_ok=True, parents=True)
    for safe_dir in (reference_safe_dir, secondary_safe_dir):
        log.info(f'Downloading orbit file for {safe_dir}')
        orbit_file = fetch_for_scene(safe_dir.stem, dir=orbit_dir)
        log.info(f'Got orbit file {orbit_file} from s1_orbits')

    config = topsapp.TopsappConfig(
        reference_safe=reference_safe_dir.name,
        secondary_safe=secondary_safe_dir.name,
        polarization=polarization,
        orbit_directory=str(orbit_dir),
        aux_cal_directory=str(aux_cal_dir),
        dem_filename=str(dem_path),
        geocode_dem_filename=str(geocode_dem_path),
        roi=roi,
        swaths=swaths,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )
    config_path = config.write_template('topsApp.xml')

    if apply_water_mask:
        topsapp.run_topsapp(start='startup', end='filter', config_xml=config_path)
        water_mask_path = 'water_mask.wgs84'
        create_water_mask(str(dem_path), water_mask_path)
        multilook(
            'merged/lon.rdr.full',
            outname='merged/lon.rdr',
            alks=azimuth_looks,
            rlks=range_looks,
        )
        multilook(
            'merged/lat.rdr.full',
            outname='merged/lat.rdr',
            alks=azimuth_looks,
            rlks=range_looks,
        )
        resample_to_radar_io(water_mask_path, 'merged/lat.rdr', 'merged/lon.rdr', 'merged/water_mask.rdr')
        isce2_copy('merged/phsig.cor', 'merged/unmasked.phsig.cor')
        image_math(
            'merged/unmasked.phsig.cor',
            'merged/water_mask.rdr',
            'merged/phsig.cor',
            'a*b',
        )
        topsapp.run_topsapp(start='unwrap', end='unwrap2stage', config_xml=config_path)
        isce2_copy('merged/unmasked.phsig.cor', 'merged/phsig.cor')
    else:
        topsapp.run_topsapp(start='startup', end='unwrap2stage', config_xml=config_path)
    copyfile('merged/z.rdr.full.xml', 'merged/z.rdr.full.vrt.xml')
    topsapp.run_topsapp(start='geocode', end='geocode', config_xml=config_path)

    return Path('merged')
