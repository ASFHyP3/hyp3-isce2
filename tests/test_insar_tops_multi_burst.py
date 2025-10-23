import pytest

from hyp3_isce2 import insar_tops_multi_burst


REF_SEC_GRANULES_ERROR = r'^Expected either --reference and --secondary or --granules$'

TWO_GRANULES_ERROR = r'^--granules must specify exactly two granules$'


# TODO:
#  - other errors?
#  - other nested list format
#  - other function behavior
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
