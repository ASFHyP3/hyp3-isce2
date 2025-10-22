import sys

import pytest

from hyp3_isce2 import insar_tops_multi_burst


def test_granules_cli(monkeypatch):
    error_str = r'^Expected either --reference and --secondary or --granules$'

    monkeypatch.setattr(sys, 'argv', ['cmd', '--reference', 'foo'])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()

    monkeypatch.setattr(sys, 'argv', ['cmd', '--secondary', 'foo'])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()

    monkeypatch.setattr(sys, 'argv', ['cmd', '--reference', 'foo', '--granules', 'foo'])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()

    monkeypatch.setattr(sys, 'argv', ['cmd', '--secondary', 'foo', '--granules', 'foo'])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()

    monkeypatch.setattr(sys, 'argv', ['cmd', '--reference', 'foo', '--secondary', 'foo', '--granules', 'foo'])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()

    error_str = '^--granules must specify exactly two granules$'

    monkeypatch.setattr(sys, 'argv', ['cmd', '--granules', ''])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()

    monkeypatch.setattr(sys, 'argv', ['cmd', '--granules', 'foo'])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()

    monkeypatch.setattr(sys, 'argv', ['cmd', '--granules', 'foo foo foo'])
    with pytest.raises(ValueError, match=error_str):
        insar_tops_multi_burst.main()
