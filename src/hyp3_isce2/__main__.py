"""ISCE2 processing for HyP3"""

import argparse
import os
import sys
import warnings
from importlib.metadata import entry_points
from pathlib import Path

from hyp3lib.fetch import write_credentials_to_netrc_file


def main():
    """Main entrypoint for HyP3 processing

    Calls the HyP3 entrypoint specified by the `++process` argument
    """
    parser = argparse.ArgumentParser(prefix_chars='+', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '++process',
        choices=[
            'insar_tops_burst',
            'insar_tops',
            'insar_stripmap',
            'merge_tops_bursts',
        ],
        default='insar_tops_burst',
        help='Select the HyP3 entrypoint to use',  # HyP3 entrypoints are specified in `pyproject.toml`
    )
    parser.add_argument(
        '++omp-num-threads',
        type=int,
        help='The number of OpenMP threads to use for parallel regions',
    )

    args, unknowns = parser.parse_known_args()

    username = os.getenv('EARTHDATA_USERNAME')
    password = os.getenv('EARTHDATA_PASSWORD')
    if username and password:
        write_credentials_to_netrc_file(username, password, append=False)

    if not (Path.home() / '.netrc').exists():
        warnings.warn(
            'Earthdata credentials must be present as environment variables, or in your netrc.',
            UserWarning,
        )

    # NOTE: Cast to set because of: https://github.com/pypa/setuptools/issues/3649
    # NOTE: Will need to update to `entry_points(group='hyp3', name=args.process)` when updating to python 3.10
    eps = entry_points()['hyp3']
    (process_entry_point,) = {process for process in eps if process.name == args.process}

    if args.omp_num_threads:
        os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)

    sys.argv = [args.process, *unknowns]
    sys.exit(process_entry_point.load()())


if __name__ == '__main__':
    main()
