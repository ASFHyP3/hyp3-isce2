"""Tests for hyp3_isce2.merge_tops_bursts module, use single quotes"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import asf_search
import isceobj  # noqa: I100
import lxml.etree as ET
import numpy as np
import pytest
from osgeo import gdal, osr
from requests import Session

import hyp3_isce2.burst as burst_utils
import hyp3_isce2.merge_tops_bursts as merge
from hyp3_isce2 import utils


def mock_asf_search_results(
    slc_name: str,
    subswath: str,
    polarization: str,
    burst_index: int,
    burst_id: int,
    path_number: int,
) -> asf_search.ASFSearchResults:
    product = asf_search.ASFProduct()
    product.umm = {
        "InputGranules": [slc_name],
        "TemporalExtent": {
            "RangeDateTime": {"BeginningDateTime": "2020-06-04T02:23:13.963847Z"}
        },
    }
    product.properties.update(
        {
            "burst": {
                "subswath": subswath,
                "burstIndex": burst_index,
                "relativeBurstID": burst_id,
            },
            "polarization": polarization,
            "url": f"https://foo.com/{slc_name}/baz.zip",
            "pathNumber": path_number,
        }
    )
    results = asf_search.ASFSearchResults([product])
    results.searchComplete = True
    return results


def create_test_geotiff(output_file, dtype="float", n_bands=1):
    """Create a test geotiff for testing"""
    opts = {
        "float": (np.float64, gdal.GDT_Float64),
        "cfloat": (np.complex64, gdal.GDT_CFloat32),
    }
    np_dtype, gdal_dtype = opts[dtype]
    data = np.ones((10, 10), dtype=np_dtype)
    geotransform = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_file, 10, 10, n_bands, gdal_dtype)
    dataset.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    for i in range(n_bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(data)
    dataset = None


def test_to_burst_params(burst_product1):
    assert burst_product1.to_burst_params() == burst_utils.BurstParams(
        "S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85",
        "IW2",
        "VV",
        7,
    )


def test_get_burst_metadata(test_merge_dir, burst_product1):
    product_path = list(test_merge_dir.glob("S1_136231*"))[0]

    with patch("hyp3_isce2.merge_tops_bursts.asf_search.granule_search") as mock_search:
        mock_search.return_value = mock_asf_search_results(
            "S1A_IW_SLC__1SDV_20200604T022251_20200604T022318_032861_03CE65_7C85",
            "IW2",
            "VV",
            7,
            136231,
            64,
        )
        product = merge.get_burst_metadata([product_path])[0]

    assert product == burst_product1


def test_prep_metadata_dirs(tmp_path):
    annotation_dir, manifest_dir = merge.prep_metadata_dirs(tmp_path)
    assert annotation_dir.is_dir()
    assert manifest_dir.is_dir()


def test_download_metadata_xmls(monkeypatch, tmp_path, test_data_dir):
    params = [
        burst_utils.BurstParams("foo", "IW1", "VV", 1),
        burst_utils.BurstParams("foo", "IW2", "VV", 0),
    ]
    sample_xml = ET.parse(test_data_dir / "reference_descending.xml").getroot()

    with patch(
        "hyp3_isce2.merge_tops_bursts.burst_utils.get_asf_session"
    ) as mock_session:
        with patch(
            "hyp3_isce2.merge_tops_bursts.burst_utils.download_metadata"
        ) as mock_download:
            mock_session.return_value = Session()
            mock_download.return_value = sample_xml

            merge.download_metadata_xmls(params, tmp_path)

            assert mock_download.call_count == 1
            assert len(list((tmp_path / "annotation").glob("*.xml"))) == 2
            assert (tmp_path / "manifest" / "foo.xml").exists()


def test_get_scene_roi(test_s1_obj):
    bursts = test_s1_obj.product.bursts
    roi = merge.get_scene_roi(bursts)
    golden_roi = (
        53.045079513806,
        27.325111859227817,
        54.15684468161031,
        27.847161580403135,
    )
    assert np.all(np.isclose(roi, golden_roi))


def test_load_isce_s1_obj(annotation_manifest_dirs):
    annotation_dir, manifest_dir = annotation_manifest_dirs
    s1_obj = merge.load_isce_s1_obj(2, "VV", annotation_dir.parent)

    assert isinstance(s1_obj, merge.Sentinel1BurstSelect)
    assert s1_obj.swath == 2
    assert s1_obj.polarization == "vv"
    assert len(s1_obj.tiff) == 1
    assert s1_obj.tiff[0] == ""


def test_Sentinel1BurstSelect(annotation_manifest_dirs, tmp_path, burst_product1):
    annotation_dir, manifest_dir = annotation_manifest_dirs
    s1_obj = merge.load_isce_s1_obj(2, "VV", annotation_dir.parent)

    # Test select_bursts
    test1_obj = deepcopy(s1_obj)
    test1_utc = [burst_product1.start_utc]
    test1_obj.select_bursts(test1_utc)
    assert len(test1_obj.product.bursts) == 1
    assert test1_obj.product.numberOfBursts == 1
    assert test1_obj.product.bursts[0].burstStartUTC == test1_utc[0]

    test2_obj = deepcopy(s1_obj)
    test2_utc = [
        datetime(2020, 6, 4, 2, 22, 57, 414185),
        datetime(2020, 6, 4, 2, 22, 54, 655908),
    ]
    test2_obj.select_bursts(test2_utc)
    assert len(test2_obj.product.bursts) == 2
    assert (
        test2_obj.product.bursts[0].burstStartUTC
        < test2_obj.product.bursts[1].burstStartUTC
    )

    # Test update_burst_properties
    test3_obj = deepcopy(test1_obj)
    outpath = tmp_path / "IW2"
    test3_obj.output = str(outpath)
    test3_obj.update_burst_properties([burst_product1])
    assert test3_obj.product.bursts[0].burstNumber == 1
    assert test3_obj.product.bursts[0].firstValidLine == 8
    assert test3_obj.product.bursts[0].numValidLines == 363
    assert test3_obj.product.bursts[0].firstValidSample == 9
    assert test3_obj.product.bursts[0].numValidSamples == 1220
    assert Path(test3_obj.product.bursts[0].image.filename).name == "burst_01.slc"

    test4_obj = deepcopy(test1_obj)
    bad_product = deepcopy(burst_product1)
    bad_product.start_utc = datetime(1999, 1, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError, match=".*do not match.*"):
        test4_obj.update_burst_properties([bad_product])

    # Test write_xml
    test5_obj = deepcopy(test3_obj)
    test5_obj.write_xml()
    assert outpath.with_suffix(".xml").exists()


def test_create_burst_cropped_s1_obj(annotation_manifest_dirs, burst_product1):
    s1_obj = merge.create_burst_cropped_s1_obj(
        2, [burst_product1], "VV", base_dir=annotation_manifest_dirs[0].parent
    )
    assert isinstance(s1_obj, merge.Sentinel1BurstSelect)
    assert Path(s1_obj.output).with_suffix(".xml").exists()


def test_modify_for_multilook(annotation_manifest_dirs, burst_product1):
    burst_product = burst_product1
    s1_obj = merge.create_burst_cropped_s1_obj(
        2, [burst_product], "VV", base_dir=annotation_manifest_dirs[0].parent
    )

    pre_burst = s1_obj.product.bursts[0]
    assert not pre_burst.numberOfSamples == burst_product.n_samples
    assert not pre_burst.numberOfLines == burst_product.n_lines

    burst_product.isce2_burst_number = s1_obj.product.bursts[0].burstNumber
    looked_obj = merge.modify_for_multilook([burst_product], s1_obj)
    burst = looked_obj.product.bursts[0]
    assert burst.numberOfSamples == burst_product.n_samples
    assert burst.numberOfLines == burst_product.n_lines
    assert burst.firstValidSample == burst_product.first_valid_sample
    assert burst.numValidSamples == burst_product.n_valid_samples
    assert burst.firstValidLine == burst_product.first_valid_line
    assert burst.numValidLines == burst_product.n_valid_lines
    assert burst.sensingStop == burst_product.stop_utc
    assert burst.azimuthTimeInterval == burst_product.az_time_interval
    assert burst.rangePixelSize == burst_product.rg_pixel_size
    assert looked_obj.output == "fine_interferogram/IW2_multilooked"

    bad_product = deepcopy(burst_product)
    bad_product.start_utc = datetime(1999, 1, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError, match=".*do not match.*"):
        looked_obj = merge.modify_for_multilook([bad_product], s1_obj)


def test_download_dem_for_multiple_bursts(annotation_manifest_dirs, burst_product1):
    base_dir = annotation_manifest_dirs[0].parent
    s1_obj = merge.create_burst_cropped_s1_obj(
        2, [burst_product1], "VV", base_dir=base_dir
    )
    with patch("hyp3_isce2.merge_tops_bursts.download_dem_for_isce2") as mock_download:
        mock_download.return_value = None
        merge.download_dem_for_multiple_bursts([s1_obj])
        assert mock_download.call_count == 1
        assert isinstance(mock_download.call_args[0][0], tuple)
        assert len(mock_download.call_args[0][0]) == 4
        assert mock_download.call_args[1]["dem_name"] == "glo_30"


@pytest.mark.parametrize(
    "isce_type,dtype,n_bands",
    [["ifg", "cfloat", 1], ["lat", "float", 1], ["los", "float", 2]],
)
def test_translate_image(isce_type, dtype, n_bands, tmp_path):
    test_tiff = tmp_path / "test.tif"
    create_test_geotiff(str(test_tiff), dtype, n_bands)
    out_path = tmp_path / "test.bin"
    merge.translate_image(str(test_tiff), str(out_path), isce_type)
    for ext in [".xml", ".vrt", ""]:
        assert (out_path.parent / (out_path.name + ext)).exists()

    opts = {"float": np.float32, "cfloat": np.complex64}
    image, array = utils.load_isce2_image(str(out_path))
    assert np.all(array == np.ones((10, 10), dtype=opts[dtype]))


def test_spoof_isce2_setup(annotation_manifest_dirs, burst_product1):
    tmp_product = deepcopy(burst_product1)
    tmp_product.isce2_burst_number = 1
    base_dir = annotation_manifest_dirs[0].parent
    s1_obj = merge.create_burst_cropped_s1_obj(
        2, [tmp_product], "VV", base_dir=base_dir
    )
    merge.spoof_isce2_setup([tmp_product], base_dir=base_dir)

    fine_ifg_dir = base_dir / "fine_interferogram" / "IW2"
    assert fine_ifg_dir.is_dir()
    assert len(list(fine_ifg_dir.glob("*"))) == 3

    geom_ref_dir = base_dir / "geom_reference" / "IW2"
    assert geom_ref_dir.is_dir()
    assert len(list(geom_ref_dir.glob("*"))) == 9


def test_get_swath_list(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    assert merge.get_swath_list(str(test_dir)) == []

    for x in [1, 2, 3]:
        (test_dir / f"IW{x}").mkdir()
    assert merge.get_swath_list(str(test_dir)) == [1, 2, 3]


def test_get_merged_orbit(test_s1_obj):
    merged_orbit = merge.get_merged_orbit([test_s1_obj.product])
    assert isinstance(merged_orbit, isceobj.Orbit.Orbit.Orbit)
    assert (
        len(merged_orbit.stateVectors) == 17
    )  # This number will change if the test data changes


def test_get_frames_and_indexes(isce2_merge_setup):
    frames, burst_index = merge.get_frames_and_indexes(
        isce2_merge_setup / "fine_interferogram"
    )
    assert len(frames) == 1
    assert isinstance(
        frames[0], isceobj.Sensor.TOPS.TOPSSwathSLCProduct.TOPSSwathSLCProduct
    )
    assert burst_index[0] == [2, 0, 2]


# FIX: test_merge_bursts doesn't work due to pathing issue.
# def test_merge_bursts(isce2_merge_setup):
#     import os
#     os.chdir(isce2_merge_setup)
#     merge.merge_bursts(20, 4)


def test_goldstein_werner_filter(tmp_path):
    in_path = tmp_path / "test.bin"
    coh_path = tmp_path / "coh.bin"
    out_path = tmp_path / "filtered.bin"
    array = np.ones((10, 10), dtype=np.complex64)
    utils.write_isce2_image(str(in_path), array)
    utils.write_isce2_image(str(coh_path), array.astype(np.float32))
    merge.goldstein_werner_filter(str(in_path), str(out_path), str(coh_path))
    assert out_path.exists()
    assert (out_path.parent / f"{out_path.name}.xml").exists()
    assert (out_path.parent / f"{out_path.name}.vrt").exists()


def test_get_product_name(burst_product1):
    product_name = merge.get_product_name(burst_product1, 80)
    assert len(product_name) == 39
    assert product_name[:-4] == "S1_064__20200604_20200616_VV_INT80_"


def test_make_parameter_file(test_data_dir, test_merge_dir, test_s1_obj, tmp_path):
    ifg_dir = tmp_path / "fine_interferogram" / "IW2"
    ifg_dir.mkdir(parents=True)
    test_s1_obj.output = str(ifg_dir.parent / "IW2")
    test_s1_obj.write_xml()

    metas = merge.get_product_metadata_info(test_merge_dir)

    out_file = tmp_path / "test.txt"
    merge.make_parameter_file(out_file, metas, 20, 4, 0.6, True, base_dir=tmp_path)
    assert out_file.exists()

    meta = utils.read_product_metadata(out_file)
    assert len(meta["ReferenceGranule"].split(",")) == 2
    assert len(meta["SecondaryGranule"].split(",")) == 2
    assert meta["Rangelooks"] == "20"
    assert meta["Azimuthlooks"] == "4"
    with pytest.raises(KeyError):
        assert meta["Radarnlines"]


def test_snaphu_unwrap(test_s1_obj, test_data_dir, tmp_path):
    merge_dir = tmp_path / "merged"
    merge_dir.mkdir()
    ifg_dir = tmp_path / "fine_interferogram" / "IW2"
    ifg_dir.mkdir(parents=True, exist_ok=True)
    test_s1_obj.output = str(ifg_dir.parent / "IW2_multilooked")
    test_s1_obj.write_xml()

    filt_path = merge_dir / "filt_topophase.flat"
    coh_path = merge_dir / "coh.bin"
    array = np.ones((100, 100), dtype=np.complex64)
    utils.write_isce2_image(str(filt_path), array)
    utils.write_isce2_image(str(coh_path), array.astype(np.float32))
    merge.snaphu_unwrap(2, 2, str(coh_path), base_dir=merge_dir)

    assert (merge_dir / "filt_topophase.unw").exists()
    assert (merge_dir / "filt_topophase.unw.xml").exists()
    assert (merge_dir / "filt_topophase.unw.vrt").exists()


def test_geocode_products(test_data_dir, tmp_path, test_s1_obj):
    merge_dir = tmp_path / "merged"
    merge_dir.mkdir()
    ifg_dir = tmp_path / "fine_interferogram" / "IW2"
    ifg_dir.mkdir(parents=True, exist_ok=True)
    test_s1_obj.output = str(ifg_dir.parent / "IW2")
    test_s1_obj.write_xml()

    unw_path = merge_dir / "filt_topophase.unw"
    dem_path = merge_dir / "dem.bin"
    array = np.ones((377, 1272), dtype=np.float32)
    utils.write_isce2_image(str(unw_path), array)
    utils.write_isce2_image(str(dem_path), array)
    merge.geocode_products(
        1, 1, dem_path, base_dir=merge_dir, to_be_geocoded=[str(unw_path)]
    )

    assert (merge_dir / "filt_topophase.unw.geo").exists()
    assert (merge_dir / "filt_topophase.unw.geo.xml").exists()
    assert (merge_dir / "filt_topophase.unw.geo.vrt").exists()


def test_check_burst_group_validity():
    @dataclass
    class Product:
        reference_date: datetime
        secondary_date: datetime
        polarization: str
        relative_orbit: int
        swath: int
        burst_id: int
        range_looks: int
        azimuth_looks: int

    # Test valid products
    good_products = [
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 1, 111116, 20, 4),
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 1, 111117, 20, 4),
    ]
    merge.check_burst_group_validity(good_products)

    # Bad polarization
    bad_pol = deepcopy(good_products)
    bad_pol[0].polarization = "HH"
    with pytest.raises(ValueError, match="All products.*polarization.*"):
        merge.check_burst_group_validity(bad_pol)

    # Bad reference date
    bad_ref_date = deepcopy(good_products)
    bad_ref_date[1].reference_date = datetime(2020, 2, 2)
    with pytest.raises(ValueError, match="All products.*reference date.*"):
        merge.check_burst_group_validity(bad_ref_date)

    # Bad reference date
    bad_sec_date = deepcopy(good_products)
    bad_sec_date[1].secondary_date = datetime(2020, 2, 2)
    with pytest.raises(ValueError, match="All products.*secondary date.*"):
        merge.check_burst_group_validity(bad_sec_date)

    # Bad relative orbit
    bad_rel_orbit = deepcopy(good_products)
    bad_rel_orbit[0].relative_orbit = 2
    with pytest.raises(ValueError, match="All products.*relative orbit.*"):
        merge.check_burst_group_validity(bad_rel_orbit)

    # Bad looks
    bad_range_looks = deepcopy(good_products)
    bad_range_looks[1].range_looks = 10
    with pytest.raises(ValueError, match="All products.*looks.*"):
        merge.check_burst_group_validity(bad_range_looks)

    # Non-contiguous bursts
    non_contiguous_swath_products = [
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 1, 111115, 20, 4),
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 1, 111117, 20, 4),
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 2, 111115, 20, 4),
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 2, 111116, 20, 4),
    ]
    with pytest.raises(ValueError, match="Products.*swath 1.*contiguous"):
        merge.check_burst_group_validity(non_contiguous_swath_products)

    # Non-contiguous swath bursts
    non_contiguous_swath_products = [
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 1, 111116, 20, 4),
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 1, 111117, 20, 4),
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 2, 111114, 20, 4),
        Product(datetime(2020, 2, 1), datetime(2020, 1, 1), "VV", 1, 2, 111113, 20, 4),
    ]
    with pytest.raises(ValueError, match="Products.*swaths 1 and 2.*overlap"):
        merge.check_burst_group_validity(non_contiguous_swath_products)


def test_get_product_multilook(tmp_path):
    product_dir = tmp_path / "test"
    product_dir.mkdir()
    product1 = product_dir / "S1_111111_IW1_VV_01"
    product1.mkdir()
    metadata1 = product1 / "S1_111111_IW1_VV_01.txt"

    metadata1.write_text("Rangelooks: 20\nAzimuthlooks: 4\n")
    range_looks, azimuth_looks = merge.get_product_multilook(product_dir)
    assert range_looks == 20
    assert azimuth_looks == 4


def test_make_readme(tmp_path):
    prod_name = "foo"
    tmp_prod_dir = tmp_path / prod_name
    tmp_prod_dir.mkdir(exist_ok=True)
    create_test_geotiff(str(tmp_prod_dir / f"{prod_name}_wrapped_phase.tif"))
    reference_scenes = ["a_a_a_20200101T000000_a", "b_b_b_20200101T000000_b"]
    secondary_scenes = ["c_c_c_20210101T000000_c", "d_d_d_20210101T000000_d"]

    merge.make_readme(tmp_prod_dir, reference_scenes, secondary_scenes, 2, 10, True)
    out_path = tmp_prod_dir / f"{prod_name}_README.md.txt"
    assert out_path.exists()
