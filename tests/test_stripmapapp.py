import pytest

from hyp3_isce2.stripmapapp_alos import StripmapappConfig, run_stripmapapp


def test_stripmap_config(tmp_path):
    config = StripmapappConfig(
        reference_image='ALPSRP156121200-L1.0/IMG-HH-ALPSRP156121200-H1.0__A',
        reference_leader='ALPSRP156121200-L1.0/LED-ALPSRP156121200-H1.0__A',
        secondary_image='ALPSRP162831200-L1.0/IMG-HH-ALPSRP162831200-H1.0__A',
        secondary_leader='ALPSRP162831200-L1.0/LED-ALPSRP162831200-H1.0__A',
        roi=[-153.212, 59.96148524563291, -151.871, 60.56159446867566],
        dem_filename='dem/full_res.dem.wgs84',
    )

    template_path = tmp_path / 'stripmapApp.xml'
    config.write_template(template_path)
    assert template_path.exists()

    with open(template_path, 'r') as template_file:
        template = template_file.read()
        assert 'ALPSRP156121200-L1.0/IMG-HH-ALPSRP156121200-H1.0__A' in template
        assert 'ALPSRP156121200-L1.0/LED-ALPSRP156121200-H1.0__A' in template
        assert 'ALPSRP162831200-L1.0/IMG-HH-ALPSRP162831200-H1.0__A' in template
        assert 'ALPSRP162831200-L1.0/LED-ALPSRP162831200-H1.0__A' in template
        assert 'dem/full_res.dem.wgs84' in template
        assert '[59.96148524563291, 60.56159446867566, -153.212, -151.871]' in template


def test_run_stripmapapp(tmp_path):
    with pytest.raises(IOError):
        run_stripmapapp('stripmapApp.xml')

    config = StripmapappConfig(
        reference_image='',
        reference_leader='',
        secondary_image='',
        secondary_leader='',
        roi=[1, 2, 3, 4],
        dem_filename='',
    )
    template_path = config.write_template(tmp_path / 'stripmapApp.xml')

    with pytest.raises(ValueError, match=r'.*not a valid step.*'):
        run_stripmapapp('notastep', config_xml=template_path)

    with pytest.raises(ValueError, match=r'^If dostep is specified, start and stop cannot be used$'):
        run_stripmapapp('preprocess', 'startup', config_xml=template_path)
