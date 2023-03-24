def test_hyp3_isce2(script_runner):
    ret = script_runner.run('python', '-m', 'hyp3_isce2', '-h')
    assert ret.success
