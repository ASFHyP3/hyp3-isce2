"""Tests for the single-burst specific functionality found in burst.py"""

from datetime import datetime

import pytest

from hyp3_isce2 import burst


def test_num_swath_pol():
    assert burst._num_swath_pol('S1_136231_IW2_20200604T022312_VV_7C85-BURST') == '136231_IW2_VV'
    assert burst._num_swath_pol('S1_068687_IW3_20230423T223824_HH_BA77-BURST') == '068687_IW3_HH'


def test_burst_datetime():
    assert burst._burst_datetime('S1_136231_IW2_20200604T022312_VV_7C85-BURST') == datetime(2020, 6, 4, 2, 23, 12)
    assert burst._burst_datetime('S1_068687_IW3_20230423T223824_HH_BA77-BURST') == datetime(2023, 4, 23, 22, 38, 24)
    assert burst._burst_datetime('S1_068687_IW3_20230403T020804_HH_BA77-BURST') == datetime(2023, 4, 3, 2, 8, 4)


def test_validate_bursts_list_length():
    burst.validate_bursts(
        ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        ['S1_000000_IW1_20200201T000000_VV_0000-BURST'],
    )

    with pytest.raises(ValueError, match=r'^Must include at least 1 reference scene and 1 secondary scene$'):
        burst.validate_bursts(['a'], [])

    with pytest.raises(ValueError, match=r'^Must include at least 1 reference scene and 1 secondary scene$'):
        burst.validate_bursts([], ['a'])

    with pytest.raises(ValueError, match=r'^Must include at least 1 reference scene and 1 secondary scene$'):
        burst.validate_bursts([], [])

    with pytest.raises(
        ValueError,
        match=r'^Must provide the same number of reference and secondary scenes, got 2 reference and 1 secondary$',
    ):
        burst.validate_bursts(['a', 'b'], ['c'])


def test_validate_bursts_num_swath_pol():
    burst.validate_bursts(
        ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        ['S1_000000_IW1_20200201T000000_VV_0000-BURST'],
    )
    burst.validate_bursts(
        [
            'S1_000000_IW1_20200101T000000_VV_0000-BURST',
            'S1_000001_IW2_20200101T000001_VV_0000-BURST',
        ],
        [
            'S1_000001_IW2_20200201T000001_VV_0000-BURST',
            'S1_000000_IW1_20200201T000000_VV_0000-BURST',
        ],
    )

    with pytest.raises(
        ValueError,
        match=r'^Burst number \+ swath \+ polarization identifiers must be the same for reference scenes and secondary scenes$',
    ):
        # Different burst number
        burst.validate_bursts(
            [
                'S1_000001_IW2_20200101T000001_VV_0000-BURST',
                'S1_000000_IW1_20200101T000000_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20200201T000000_VV_0000-BURST',
                'S1_000002_IW2_20200201T000001_VV_0000-BURST',
            ],
        )

    with pytest.raises(
        ValueError,
        match=r'^Burst number \+ swath \+ polarization identifiers must be the same for reference scenes and secondary scenes$',
    ):
        # Different swath
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
            ['S1_000000_IW2_20200201T000000_VV_0000-BURST'],
        )

    with pytest.raises(
        ValueError,
        match=r'^Burst number \+ swath \+ polarization identifiers must be the same for reference scenes and secondary scenes$',
    ):
        # Different pol
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
            ['S1_000000_IW1_20200201T000000_VH_0000-BURST'],
        )

    with pytest.raises(
        ValueError,
        match=r'^Each reference scene must have a unique burst number \+ swath \+ polarization identifier$',
    ):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20200101T000000_VV_0000-BURST',
                'S1_000000_IW1_20200101T000001_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20200201T000000_VV_0000-BURST',
                'S1_000000_IW1_20200201T000002_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Scenes must have the same polarization. Polarizations present: VH, VV$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20200101T000000_VH_0000-BURST',
                'S1_000000_IW1_20200101T000000_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20200201T000000_VV_0000-BURST',
                'S1_000000_IW1_20200201T000000_VH_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^VH polarization is not currently supported, only VV and HH$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20200101T000000_VH_0000-BURST',
                'S1_000000_IW2_20200101T000000_VH_0000-BURST',
            ],
            [
                'S1_000000_IW1_20200201T000000_VH_0000-BURST',
                'S1_000000_IW2_20200201T000000_VH_0000-BURST',
            ],
        )


def test_validate_bursts_datetimes():
    burst.validate_bursts(
        ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        ['S1_000000_IW1_20200101T000001_VV_0000-BURST'],
    )
    burst.validate_bursts(
        [
            'S1_000000_IW1_20250101T000000_VV_0000-BURST',
            'S1_000001_IW1_20250101T000100_VV_0000-BURST',
            'S1_000002_IW1_20250101T000200_VV_0000-BURST',
        ],
        [
            'S1_000000_IW1_20250101T000201_VV_0000-BURST',
            'S1_000001_IW1_20250101T000300_VV_0000-BURST',
            'S1_000002_IW1_20250101T000401_VV_0000-BURST',
        ],
    )

    with pytest.raises(ValueError, match=r'^Reference scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000201_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000401_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                # Test with reference datetimes unsorted
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000401_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Secondary scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000402_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Secondary scenes must fall within a 2-minute window .*'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                # Test with secondary datetimes unsorted
                'S1_000000_IW1_20250101T000402_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000201_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            ['S1_000000_IW1_20200101T000001_VV_0000-BURST'],
            ['S1_000000_IW1_20200101T000000_VV_0000-BURST'],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000200_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000400_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000201_VV_0000-BURST',
                'S1_000001_IW1_20250101T000300_VV_0000-BURST',
                'S1_000002_IW1_20250101T000401_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000000_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
        )

    with pytest.raises(ValueError, match=r'^Reference scenes must be older than secondary scenes$'):
        # Test with reference and secondary datetimes unsorted
        burst.validate_bursts(
            [
                'S1_000000_IW1_20250101T000200_VV_0000-BURST',
                'S1_000001_IW1_20250101T000100_VV_0000-BURST',
                'S1_000002_IW1_20250101T000000_VV_0000-BURST',
            ],
            [
                'S1_000000_IW1_20250101T000300_VV_0000-BURST',
                'S1_000001_IW1_20250101T000400_VV_0000-BURST',
                'S1_000002_IW1_20250101T000200_VV_0000-BURST',
            ],
        )
