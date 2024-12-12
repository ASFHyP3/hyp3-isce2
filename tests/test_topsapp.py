import pytest

from hyp3_isce2.topsapp import TopsappConfig, run_topsapp, swap_burst_vrts


def test_topsapp_burst_config(tmp_path):
    config = TopsappConfig(
        reference_safe="S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE",
        secondary_safe="S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11.SAFE",
        polarization="VV",
        orbit_directory="orbits",
        aux_cal_directory="aux_cal",
        roi=[-118.0, 37.0, -117.0, 38.0],
        dem_filename="dem.tif",
        geocode_dem_filename="dem_geocode.tif",
        swaths=1,
    )

    template_path = tmp_path / "topsapp.xml"
    config.write_template(template_path)
    assert template_path.exists()

    with open(template_path, "r") as template_file:
        template = template_file.read()
        assert (
            "S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85.SAFE"
            in template
        )
        assert (
            "S1A_IW_SLC__1SDV_20200616T022252_20200616T022319_033036_03D3A3_5D11.SAFE"
            in template
        )
        assert "orbits" in template
        assert "aux_cal" in template
        assert "dem.tif" in template
        assert "dem_geocode.tif" in template
        assert "[37.0, 38.0, -118.0, -117.0]" in template
        assert "[1]" in template


def test_swap_burst_vrts(tmp_path, monkeypatch):
    ref_vrt_dir = tmp_path / "reference" / "tmp"
    ref_vrt_dir.mkdir(parents=True)
    (ref_vrt_dir / "reference.vrt").touch()

    sec_vrt_dir = tmp_path / "secondary" / "tmp"
    sec_vrt_dir.mkdir(parents=True)
    (sec_vrt_dir / "secondary.vrt").touch()
    (sec_vrt_dir / "bad.vrt").touch()

    monkeypatch.chdir(str(tmp_path))
    with pytest.raises(ValueError, match=r"There should only be 2 VRT files .*"):
        swap_burst_vrts()


def test_run_topsapp_burst(tmp_path, monkeypatch):
    with pytest.raises(IOError):
        run_topsapp("topsApp.xml")

    config = TopsappConfig(
        reference_safe="",
        secondary_safe="",
        polarization="",
        orbit_directory="",
        aux_cal_directory="",
        roi=[0, 1, 2, 3],
        dem_filename="",
        geocode_dem_filename="",
        swaths=1,
        azimuth_looks=1,
        range_looks=1,
    )
    template_path = config.write_template(tmp_path / "topsApp.xml")

    with pytest.raises(ValueError, match=r".*not a valid step.*"):
        run_topsapp("notastep", config_xml=template_path)

    with pytest.raises(
        ValueError, match=r"^If dostep is specified, start and stop cannot be used$"
    ):
        run_topsapp("preprocess", "startup", config_xml=template_path)

    monkeypatch.chdir(tmp_path)
    run_topsapp("preprocess", config_xml=template_path)
