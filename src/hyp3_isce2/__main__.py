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

    process_entry_point = list(entry_points(group='hyp3', name=args.process))[0]

    if args.omp_num_threads:
        os.environ['OMP_NUM_THREADS'] = str(args.omp_num_threads)

    sys.argv = [args.process, *unknowns]
    sys.exit(process_entry_point.load()())


if __name__ == '__main__':
    main()
