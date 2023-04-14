def test_hyp3_isce2(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', '-h')
    assert ret.success


def test_insar_tops_burst(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', 'insart_tops_burst', '-h')
    assert ret.success


def test_insar_tops_burst_main(script_runner):
    ret = script_runner.run('insar_tops_burst', '-h')
    assert ret.success
