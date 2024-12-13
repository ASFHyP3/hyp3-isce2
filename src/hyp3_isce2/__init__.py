"""HyP3 plugin for ISCE2 processing"""

import logging
import os
from importlib.metadata import version
from pathlib import Path

# Ensures all ISCE2 paths and environment variables are set when using this module, see:
# https://github.com/isce-framework/isce2/blob/main/__init__.py#L41-L50
import isce  # noqa: F401

# ISCE2 sets the root logger to DEBUG resulting in excessively verbose logging, see:
# https://github.com/isce-framework/isce2/issues/258
root_logger = logging.getLogger()
root_logger.setLevel('WARNING')

# ISCE2 also needs its applications to be on the system path, even though they say it's only "for convenience", see:
# https://github.com/isce-framework/isce2#setup-your-environment
ISCE_APPLICATIONS = str(Path(os.environ['ISCE_HOME']) / 'applications')
if ISCE_APPLICATIONS not in (PATH := os.environ['PATH'].split(os.pathsep)):
    os.environ['PATH'] = os.pathsep.join([ISCE_APPLICATIONS] + PATH)

__version__ = version(__name__)

__all__ = [
    '__version__',
]
