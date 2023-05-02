from pathlib import Path
from typing import Iterable, Union

from isce.applications.stripmapApp import Insar
from jinja2 import Template
from osgeo import gdal

TEMPLATE_DIR = Path(__file__).parent / 'templates'
STRIPMAPAPP_STEPS = [
    'startup',
    'preprocess',
    'cropraw',
    'formslc',
    'cropslc',
    'verifyDEM',
    'topo',
    'geo2rdr',
    'coarse_resample',
    'misregistration',
    'refined_resample',
    'dense_offsets',
    'rubber_sheet_range',
    'rubber_sheet_azimuth',
    'fine_resample',
    'split_range_spectrum',
    'sub_band_resample',
    'interferogram',
    'sub_band_interferogram',
    'filter',
    'filter_low_band',
    'filter_high_band',
    'unwrap',
    'unwrap_low_band',
    'unwrap_high_band',
    'ionosphere',
    'geocode',
    'geocodeoffsets',
    'endup'
]
STRIPMAPAPP_GEOCODE_LIST = [
    'interferogram/phsig.cor',
    'interferogram/filt_topophase.unw',
    'interferogram/los.rdr',
    'interferogram/topophase.flat',
    'interferogram/filt_topophase.flat',
    'interferogram/topophase.cor',
    'interferogram/filt_topophase.unw.conncomp',
]


class StripmapappConfig:
    """Configuration for a topsApp.py run"""

    def __init__(
        self,
        reference_image: str,
        reference_leader: str,
        secondary_image: str,
        secondary_leader: str,
        roi: Iterable[float],
        dem_filename: str,
        azimuth_looks: int = 14,
        range_looks: int = 4,
        do_unwrap: bool = True,
    ):
        self.reference_image = reference_image
        self.reference_leader = reference_leader
        self.secondary_image = secondary_image
        self.secondary_leader = secondary_leader
        self.roi = [roi[1], roi[3], roi[0], roi[2]]
        self.dem_filename = dem_filename
        self.geocode_dem_filename = dem_filename
        self.azimuth_looks = azimuth_looks
        self.range_looks = range_looks

        # hardcoded params for topsapp burst processing
        self.filter_strength = 0.5
        self.filter_coherence = 0.6
        self.do_unwrap = True
        self.use_virtual_files = True
        self.geocode_list = STRIPMAPAPP_GEOCODE_LIST

    def generate_template(self) -> str:
        """Generate the topsApp.py jinja2 template

        Returns:
            The rendered template
        """
        with open(TEMPLATE_DIR / 'stripmapapp_alos.xml', 'r') as file:
            template = Template(file.read())
        return template.render(self.__dict__)

    def write_template(self, filename: Union[str, Path] = 'stripmapApp.xml') -> Path:
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

def run_stripmapapp(dostep: str = '', start: str = '', end: str = '', config_xml: Path = Path('stripmapApp.xml')):
    """Run topsApp.py for a burst pair with the desired steps and config file

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
    if not config_xml.exists():
        raise IOError(f'The config file {config_xml} does not exist!')

    if dostep and (start or end):
        raise ValueError('If dostep is specified, start and stop cannot be used')

    step_args = []
    options = {
        'dostep': dostep,
        'start': start,
        'end': end,
    }
    for key, value in options.items():
        if not value:
            continue
        if value not in STRIPMAPAPP_STEPS:
            raise ValueError(f'{value} is not a valid step')
        step_args.append(f'--{key}={value}')

    cmd_line = [str(config_xml)] + step_args
    insar = Insar(name='stripmapApp', cmdline=cmd_line)
    insar.configure()
    insar.run()
