from pathlib import Path
from typing import Iterable, Union

from isce.applications.topsApp import TopsInSAR
from jinja2 import Template

TEMPLATE_DIR = Path(__file__).parent / 'templates'
TOPSAPP_STEPS = [
    'startup',
    'preprocess',
    'computeBaselines',
    'verifyDEM',
    'topo',
    'subsetoverlaps',
    'coarseoffsets',
    'coarseresamp',
    'overlapifg',
    'prepesd',
    'esd',
    'rangecoreg',
    'fineoffsets',
    'fineresamp',
    'ion',
    'burstifg',
    'mergebursts',
    'filter',
    'unwrap',
    'unwrap2stage',
    'geocode',
    'denseoffsets',
    'filteroffsets',
    'geocodeoffsets',
]
TOPSAPP_GEOCODE_LIST = [
    'merged/phsig.cor',
    'merged/filt_topophase.unw',
    'merged/los.rdr',
    'merged/topophase.flat',
    'merged/filt_topophase.flat',
    'merged/filt_topophase_2stage.unw',
    'merged/topophase.cor',
    'merged/filt_topophase.unw.conncomp',
]


class TopsappBurstConfig:
    """Configuration for a topsApp.py run"""

    def __init__(
        self,
        reference_safe: str,
        secondary_safe: str,
        orbit_directory: str,
        aux_cal_directory: str,
        region_of_interest: Iterable[float],
        dem_filename: str,
        swath: int,
        azimuth_looks: int = 4,
        range_looks: int = 20,
        do_unwrap: bool = True,
    ):
        self.reference_safe = reference_safe
        self.secondary_safe = secondary_safe
        self.orbit_directory = orbit_directory
        self.aux_cal_directory = aux_cal_directory
        self.region_of_interest = region_of_interest
        self.dem_filename = dem_filename
        self.geocode_dem_filename = dem_filename
        self.swath = swath
        self.swaths = [self.swath]
        self.azimuth_looks = azimuth_looks
        self.range_looks = range_looks
        self.do_unwrap = do_unwrap

        # hardcoded params for topsapp burst processing
        self.estimate_ionosphere_delay = False
        self.do_esd = False
        self.esd_coherence_threshold = 0.7
        self.filter_strength = 0.5
        self.do_unwrap = True
        self.use_virtual_files = True
        self.geocode_list = TOPSAPP_GEOCODE_LIST

    def generate_template(self) -> str:
        """Generate the topsApp.py jinja2 template

        Returns:
            The rendered template
        """
        with open(TEMPLATE_DIR / 'topsapp.xml', 'r') as file:
            template = Template(file.read())
        return template.render(self.__dict__)

    def write_template(self, filename: Union[str, Path] = 'topsApp.xml') -> Path:
        """Write the topsApp.py jinja2 template to a file

        Args:
            filename: Filename to write the template to
        Returns:
            The path of the written template
        """
        if not isinstance(filename, Path):
            filename = Path(filename)

        with open(filename, 'w') as file:
            file.write(self.generate_template())

        return filename


def run_topsapp(dostep: str = '', start: str = '', stop: str = '', config_xml: str = 'topsApp.xml'):
    """Run topsApp.py with the desired steps and config file

    Args:
        dostep: The step to run
        start: The step to start at
        stop: The step to stop at
        config_xml: The config file to use

    Raises:
        ValueError: If dostep is specified, start and stop cannot be used
        IOError: If the config file does not exist
        ValueError: If the step is not a valid step (see TOPSAPP_STEPS)
    """

    if dostep and (start or stop):
        raise ValueError('If dostep is specified, start and stop cannot be used')

    if not Path(config_xml).exists():
        raise IOError(f'The config file {config_xml} does note exist!')

    step_args = []
    options = {
        'dostep': dostep,
        'start': start,
        'stop': stop,
    }
    for key, value in options.items():
        if not value:
            continue
        if value not in TOPSAPP_STEPS:
            raise ValueError(f'{value} is not a valid step')
        step_args.append(f'--{key}={value}')

    cmd_line = [config_xml] + step_args
    insar = TopsInSAR(name='topsApp', cmdline=cmd_line)
    insar.configure()
    insar.run()
