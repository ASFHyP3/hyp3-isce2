def test_hyp3_isce2(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', '-h')
    assert ret.success


def test_insar_tops_burst(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', 'insar_tops_burst', '-h')
    assert ret.success


def test_insar_stripmap(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', 'insar_stripmap', '-h')
    assert ret.success


def test_hyp3_isce2_main(script_runner):
    ret = script_runner.run('hyp3-isce2', '-h')
    assert ret.success
