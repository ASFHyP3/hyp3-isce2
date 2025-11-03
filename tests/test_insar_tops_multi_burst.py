import pytest

from hyp3_isce2 import insar_tops_multi_burst


REF_SEC_GRANULES_ERROR = r'^Expected either --reference and --secondary or --granules$'

TWO_GRANULES_ERROR = r'^--granules must specify exactly two granules$'


@pytest.mark.parametrize(
    'reference_arg,secondary_arg,granules_arg,expected_result',
    [
        ([['ref1']], [['sec1']], None, (['ref1'], ['sec1'])),
        ([['ref1', 'ref2']], [['sec1', 'sec2']], None, (['ref1', 'ref2'], ['sec1', 'sec2'])),
        ([['ref1'], ['ref2']], [['sec1'], ['sec2']], None, (['ref1', 'ref2'], ['sec1', 'sec2'])),
        (
            None,
            None,
            [['S1_136231_IW2_20200604T022312_VV_7C85-BURST', 'S1_136231_IW2_20200616T022313_VV_5D11-BURST']],
            (['S1_136231_IW2_20200604T022312_VV_7C85-BURST'], ['S1_136231_IW2_20200616T022313_VV_5D11-BURST']),
        ),
        (
            None,
            None,
            [['S1_136231_IW2_20200616T022313_VV_5D11-BURST', 'S1_136231_IW2_20200604T022312_VV_7C85-BURST']],
            (['S1_136231_IW2_20200604T022312_VV_7C85-BURST'], ['S1_136231_IW2_20200616T022313_VV_5D11-BURST']),
        ),
        (
            None,
            None,
            [['S1_136231_IW2_20200604T022312_VV_7C85-BURST'], ['S1_136231_IW2_20200616T022313_VV_5D11-BURST']],
            (['S1_136231_IW2_20200604T022312_VV_7C85-BURST'], ['S1_136231_IW2_20200616T022313_VV_5D11-BURST']),
        ),
        (
            None,
            None,
            [['S1_136231_IW2_20200616T022313_VV_5D11-BURST'], ['S1_136231_IW2_20200604T022312_VV_7C85-BURST']],
            (['S1_136231_IW2_20200604T022312_VV_7C85-BURST'], ['S1_136231_IW2_20200616T022313_VV_5D11-BURST']),
        ),
    ],
)
def test_parse_reference_secondary(reference_arg, secondary_arg, granules_arg, expected_result):
    assert (
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=reference_arg, secondary_arg=secondary_arg, granules_arg=granules_arg
        )
        == expected_result
    )


@pytest.mark.parametrize(
    'reference_arg,secondary_arg,granules_arg,error_pattern',
    [
        (None, None, None, REF_SEC_GRANULES_ERROR),
        ([['foo']], None, None, REF_SEC_GRANULES_ERROR),
        (None, [['foo']], None, REF_SEC_GRANULES_ERROR),
        ([['foo']], None, [['foo']], REF_SEC_GRANULES_ERROR),
        (None, [['foo']], [['foo']], REF_SEC_GRANULES_ERROR),
        ([['foo']], [['foo']], [['foo']], REF_SEC_GRANULES_ERROR),
        (None, None, [[]], TWO_GRANULES_ERROR),
        (None, None, [['foo']], TWO_GRANULES_ERROR),
        (None, None, [['foo', 'foo', 'foo']], TWO_GRANULES_ERROR),
        (None, None, [['foo'], ['foo'], ['foo']], TWO_GRANULES_ERROR),
    ],
)
def test_parse_reference_secondary_errors(reference_arg, secondary_arg, granules_arg, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=reference_arg, secondary_arg=secondary_arg, granules_arg=granules_arg
        )
