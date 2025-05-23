from collections.abc import Iterable, Sequence
from pathlib import Path

from isce.applications.topsApp import TopsInSAR
from jinja2 import Template
from osgeo import gdal


gdal.UseExceptions()

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
]
TOPSAPP_GEOCODE_LIST = [
    'merged/phsig.cor',
    'merged/filt_topophase.unw',
    'merged/los.rdr',
    'merged/filt_topophase.flat',
    'merged/topophase.cor',
    'merged/filt_topophase.unw.conncomp',
]


class TopsappConfig:
    """Configuration for a topsApp.py run"""

    def __init__(
        self,
        reference_safe: str,
        secondary_safe: str,
        polarization: str,
        orbit_directory: str,
        aux_cal_directory: str,
        dem_filename: str,
        geocode_dem_filename: str,
        roi: Sequence[float] | None = None,
        swaths: int | Iterable[int] = (1, 2, 3),
        azimuth_looks: int = 4,
        range_looks: int = 20,
        do_unwrap: bool = True,
    ):
        self.reference_safe = reference_safe
        self.secondary_safe = secondary_safe
        self.polarization = polarization
        self.orbit_directory = orbit_directory
        self.aux_cal_directory = aux_cal_directory
        self.dem_filename = dem_filename
        self.geocode_dem_filename = geocode_dem_filename
        self.azimuth_looks = azimuth_looks
        self.range_looks = range_looks
        self.do_unwrap = do_unwrap
        self.roi = [roi[1], roi[3], roi[0], roi[2]] if roi else roi

        if isinstance(swaths, int):
            self.swaths = [swaths]
        else:
            self.swaths = list(swaths)

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
        with open(TEMPLATE_DIR / 'topsapp.xml') as file:
            if not self.roi:
                lines = file.readlines()
                for i in range(len(lines)):
                    if 'roi' in lines[i]:
                        lines.pop(i)
                        break
                template = Template(''.join(lines))
            else:
                template = Template(file.read())
        return template.render(self.__dict__)

    def write_template(self, filename: str | Path = 'topsApp.xml') -> Path:
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


def swap_burst_vrts():
    """Swap the VRTs generated by topsApp for the reference and secondary bursts

    To convince topsApp to process a burst pair, we need to swap the VRTs it generates for the
    reference and secondary bursts with custom VRTs that point to the actual burst rasters.
    """
    ref_vrt_list = [str(path) for path in Path('reference').glob('**/*.vrt')]
    sec_vrt_list = [str(path) for path in Path('secondary').glob('**/*.vrt')]
    if len(ref_vrt_list) != 1 or len(sec_vrt_list) != 1:
        raise ValueError(
            'There should only be 2 VRT files in the reference and secondary directories, '
            'this indicates there is likely a bug in the region of interest generation.'
        )

    for vrt_path in (ref_vrt_list[0], sec_vrt_list[0]):
        vrt = gdal.Open(vrt_path)
        base = gdal.Open(vrt.GetFileList()[1])
        del vrt

        gdal.Translate(vrt_path, base, format='VRT')
        del base


def run_topsapp(
    dostep: str = '',
    start: str = '',
    end: str = '',
    config_xml: Path = Path('topsApp.xml'),
):
    """Run topsApp.py for a granule pair with the desired steps and config file

    Args:
        dostep: The step to run
        start: The step to start at
        end: The step to stop at
        config_xml: The config file to use

    Raises:
        ValueError: If dostep is specified, start and stop cannot be used
        IOError: If the config file does not exist
        ValueError: If the step is not a valid step (see TOPSAPP_STEPS)
    """
    if not config_xml.exists():
        raise OSError(f'The config file {config_xml} does not exist!')

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
        if value not in TOPSAPP_STEPS:
            raise ValueError(f'{value} is not a valid step')
        step_args.append(f'--{key}={value}')

    cmd_line = [str(config_xml)] + step_args
    insar = TopsInSAR(name='topsApp', cmdline=cmd_line)
    insar.configure()
    insar.run()
