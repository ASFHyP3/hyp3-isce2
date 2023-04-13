"""Miscellaneous examples of downloading data from the ASF Burst
Extraction API. These examples are not intended to be run as part of
the test suite, but are provided as a reference for how to use the
hyp3_isce2.burst module.

Example 1: Downloading metadata for a pair of ascending/descending bursts
            (these are the metadata files used in the test suite.)

Example 2: Downloading a pair of ascending/descending bursts and spoofing a SAFE
"""

from hyp3_isce2.burst import BurstParams, download_bursts, download_metadata, get_asf_session

ref_desc = BurstParams('S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85', 'IW2', 'VV', 3)
sec_desc = BurstParams('S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11', 'IW2', 'VV', 3)
ref_asc = BurstParams('S1A_IW_SLC__1SDV_20211229T231926_20211229T231953_041230_04E66A_3DBE', 'IW1', 'VV', 4)
sec_asc = BurstParams('S1A_IW_SLC__1SDV_20220110T231926_20220110T231953_041405_04EC57_103E', 'IW1', 'VV', 4)

# Example 1
# Download metadata files
with get_asf_session() as session:
    download_metadata(session, ref_asc, 'data/reference_ascending.xml')
    download_metadata(session, sec_desc, 'data/secondary_ascending.xml')

# Example 2
# Download with SAFE spoofing
download_bursts([ref_desc, sec_desc])
