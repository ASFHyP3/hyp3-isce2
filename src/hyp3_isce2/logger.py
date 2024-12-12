import logging
import sys


def configure_root_logger() -> None:
    """Re-configures the root level logger since ISCE2 sets it to DEBUG.
    See: https://github.com/ASFHyP3/hyp3-isce2/issues/53 for context.
    """
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )
