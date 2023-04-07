"""
ISCE2 processing
"""

import logging
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import isce
from hyp3lib.get_orb import downloadSentinelOrbitFile

from hyp3_isce2 import __version__
from hyp3_isce2.burst import BurstParams, download_bursts, get_region_of_interest
from hyp3_isce2.s1_auxcal import download_aux_cal
from hyp3_isce2.topsapp import TopsappBurstConfig


log = logging.getLogger(__name__)
TOPSAPP = str(Path(isce.__file__).parent / 'applications' / 'topsApp.py')


def topsapp_burst(
    reference_scene: str,
    secondary_scene: str,
    swath_number: int,
    reference_burst_number: int,
    secondary_burst_number: int,
    polarization: str = 'VV',
    azimuth_looks: int = 4,
    range_looks: int = 20,
) -> None:
    """Create a burst interferogram

    Args:
        reference_scene: Reference SLC name
        secondary_scene: Secondary SLC name
        swath_number: Number of swath to grab bursts from (1, 2, or 3) for IW
        reference_burst_number: Number of burst to download for reference (0-indexed from first collect)
        secondary_burst_numbe: Number of burst to download for secondary (0-indexed from first collect)
        polarization: Polarization to use
        azimuth_looks: Number of azimuth looks
        range_looks: Number of range looks
    """
    orbit_dir = Path('orbits')
    aux_cal_dir = Path('aux_cal')
    ref_params = BurstParams(reference_scene, f'IW{swath_number}', polarization.upper(), reference_burst_number)
    sec_params = BurstParams(secondary_scene, f'IW{swath_number}', polarization.upper(), secondary_burst_number)
    ref_metadata, sec_metadata = download_bursts([ref_params, sec_params])

    is_ascending = ref_metadata.orbit_direction == 'ascending'
    insar_roi = get_region_of_interest(ref_metadata.footprint, sec_metadata.footprint, is_ascending=is_ascending)
    dem_roi = ref_metadata.footprint.intersection(sec_metadata.footprint).bounds
    print(insar_roi, dem_roi)

    # TODO placeholder for downloading the DEM
    dem_filename = 'dem.tif'
    download_aux_cal(aux_cal_dir)

    orbit_dir.mkdir(exist_ok=True, parents=True)
    for granule in (ref_params.granule, sec_params.granule):
        orbit_file, _ = downloadSentinelOrbitFile(granule, orbit_dir)

    config = TopsappBurstConfig(
        reference_safe=f'{ref_params.granule}.SAFE',
        secondary_safe=f'{sec_params.granule}.SAFE',
        orbit_directory=orbit_dir,
        aux_cal_directory=aux_cal_dir,
        region_of_interest=insar_roi,
        dem_filename=dem_filename,
        swath=swath_number,
        azimuth_looks=azimuth_looks,
        range_looks=range_looks,
    )
    template_filename = config.write_template()
    print(template_filename)

    # TODO replace with the actual processing call once we have the functionality for downloading the input data
    subprocess.run(['python', TOPSAPP, '-h'])

    return None


def main():
    """process_isce2 entrypoint"""
    parser = ArgumentParser(prog='topsapp_burst', description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--reference-scene', type=str, required=True)
    parser.add_argument('--secondary-scene', type=str, required=True)
    parser.add_argument('--swath-number', type=int, required=True)
    parser.add_argument('--reference-burst-number', type=int, required=True)
    parser.add_argument('--secondary-burst-number', type=int, required=True)
    parser.add_argument('--polarization', type=str, default='VV')
    parser.add_argument('--azimuth-looks', type=int, default=4)
    parser.add_argument('--range-looks', type=int, default=20)
    args = parser.parse_args()

    topsapp_burst(**args.__dict__)


if __name__ == "__main__":
    main()
