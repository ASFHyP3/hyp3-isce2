import logging
import sys


def configure_root_logger() -> None:
    # isce2 sets the root logger level to DEBUG, so we re-configure it here.
    # See: https://github.com/ASFHyP3/hyp3-isce2/issues/53
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        force=True,
    )
