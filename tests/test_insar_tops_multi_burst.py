import pytest

from hyp3_isce2 import insar_tops_multi_burst


# TODO:
#  - other nested list format
#  - other function behavior
def test_parse_reference_secondary():
    error_str = r'^Expected either --reference and --secondary or --granules$'

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=[['foo']], secondary_arg=None, granules_arg=None
        )

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=None, secondary_arg=[['foo']], granules_arg=None
        )

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=[['foo']], secondary_arg=None, granules_arg=[['foo']]
        )

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=None, secondary_arg=[['foo']], granules_arg=[['foo']]
        )

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=[['foo']], secondary_arg=[['foo']], granules_arg=[['foo']]
        )

    error_str = '^--granules must specify exactly two granules$'

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(reference_arg=None, secondary_arg=None, granules_arg=[[]])

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=None, secondary_arg=None, granules_arg=[['foo']]
        )

    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst._parse_reference_secondary(
            reference_arg=None, secondary_arg=None, granules_arg=[['foo', 'foo', 'foo']]
        )
