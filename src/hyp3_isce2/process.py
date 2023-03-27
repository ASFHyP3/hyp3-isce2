"""
ISCE2 processing
"""

import argparse
import logging
import os
import subprocess
from typing import Iterable
from pathlib import Path

from hyp3_isce2 import __version__

log = logging.getLogger(__name__)

TOPSAPP = str(Path(os.getenv('ISCE_HOME')) / 'applications' / 'topsApp.py')


def process_topsapp_burst_ifg(arg_list: Iterable[str] = ['-h']) -> None:
    """Create a burst interferogram

    Args:
        arg_list: args to pass to topsApp.py
    """
    subprocess.run(['python', TOPSAPP] + arg_list)

    return None


def main():
    """process_isce2 entrypoint"""
    parser = argparse.ArgumentParser(
        prog='process_topsapp_burst_ifg',
        description=__doc__,
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args()

    process_topsapp_burst_ifg(**args.__dict__)


if __name__ == "__main__":
    main()
