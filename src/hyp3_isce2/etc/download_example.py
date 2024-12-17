"""Miscellaneous examples of downloading data from the ASF Burst
Extraction API. These examples are not intended to be run as part of
the test suite, but are provided as a reference for how to use the
hyp3_isce2.burst module.

Example 1: Downloading metadata for a pair of ascending/descending bursts
            (these are the metadata files used in the test suite.)

Example 2: Downloading a pair of ascending/descending bursts and spoofing a SAFE
"""

from hyp3_isce2.burst import (
    BurstParams,
    download_bursts,
    download_metadata,
    get_asf_session,
)


ref_desc = BurstParams(
    'S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85',
    'IW2',
    'VV',
    3,
)
sec_desc = BurstParams(
    'S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11',
    'IW2',
    'VV',
    3,
)
ref_asc = BurstParams(
    'S1A_IW_SLC__1SDV_20200608T142544_20200608T142610_032927_03D069_14F4',
    'IW1',
    'VV',
    1,
)
sec_asc = BurstParams(
    'S1A_IW_SLC__1SDV_20200620T142544_20200620T142611_033102_03D5B7_8F1B',
    'IW1',
    'VV',
    1,
)

# Example 1
# Download metadata files for tests
with get_asf_session() as session:
    download_metadata(session, ref_asc, 'reference_ascending.xml')
    download_metadata(session, sec_asc, 'secondary_ascending.xml')
    download_metadata(session, ref_desc, 'reference_descending.xml')
    download_metadata(session, sec_desc, 'secondary_descending.xml')

# Example 2
# Download with SAFE spoofing
download_bursts([ref_desc, sec_desc])
