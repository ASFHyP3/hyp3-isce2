def test_hyp3_isce2(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', '-h')
    assert ret.success


def test_insar_tops_burst(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', '++process', 'insar_tops_burst', '-h')
    assert ret.success


def test_insar_tops(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', '++process', 'insar_tops', '-h')
    assert ret.success


def test_insar_stripmap(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', '++process', 'insar_stripmap', '-h')
    assert ret.success


def test_insar_tops_burst_main(script_runner):
    ret = script_runner.run('insar_tops_burst', '-h')
    assert ret.success


def test_insar_tops_main(script_runner):
    ret = script_runner.run('insar_tops', '-h')
    assert ret.success


def test_insar_stripmap_main(script_runner):
    ret = script_runner.run('insar_stripmap', '-h')
    assert ret.success
