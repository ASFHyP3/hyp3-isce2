# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]
### Added
* `merge_tops_bursts.py` file and workflow for merge burst products created using insar_tops_bursts.
* `merge_tops_bursts` entrypoint
* `merge_tops_bursts` README template and creation functionality
* several classes and functions to `burst.py` and `utils.py` to support `merge_tops_burst`.
* tests for the added functionality.
* `tests/data/merge.zip` example data for testing merge workflow.
* `tests/data/create_merge_test_data.py` for generating merge workflow test data.

### Changed
* `insar_tops_burst.py` to add four radar coordinate datasets to burst InSAR products (wrapped phase, lat, lon, los).
* README files generated in `insar_tops_burst.py` are now use blocks and extends the `insar_burst_base.md.txt.j2` jinja template.

## [0.10.0]
### Added
* Support for a new water masking dataset based off of OpenStreetMaps and ESA WorldCover data.
### Removed
* Polygon processing functions: `split_geometry_on_antimeridian` and `get_envelope_wgs84` from `water_mask.py`.

## [0.9.3]
### Changed
* Upgraded to `hyp3lib=>3,<4` from `>=2,<3`
### Fixed
* @scottyhq fixed excessively verbose logging due to ISCE2 setting the root logger to `DEBUG` in [#176](https://github.com/ASFHyP3/hyp3-isce2/issues/176)

## [0.9.2]
### Fixed
* `No annotation xml file` error in `insar_tops_burst` when processing HH pairs. Fixes [#168](https://github.com/ASFHyP3/hyp3-isce2/issues/168).
### Added
* a warning that processing products over the Anti-Meridian is not currently supported.

## [0.9.1]
### Fixed
* Water Masking now pulls the global shapefile and clips it, rather than pulling a partitioned parque, due to issues around the partition boundaries.

## [0.9.0]
### Changed
* Upgraded `hyp3lib` dependency to version `2.x.x`.
* As of [HyP3-lib v2.0.0](https://github.com/ASFHyP3/hyp3-lib/releases/tag/v2.0.0), the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/) will now be used for downloading Sentinel-1 orbit files from ESA.
* CDSE credentials must be provided via the `--esa-username` and `--esa-password` command-line options, the `ESA_USERNAME` and `ESA_PASSWORD` environment variables, or a `.netrc` file.
* All the special ISCE2 environment variable, python path, and system path handling has been moved to `hyp3_isce2.__init__.py` to ensure it's always done before using any object in this package.
* All [subprocess](https://docs.python.org/3/library/subprocess.html#using-the-subprocess-module) calls use `subprocess.run`, as recommended.

## [0.8.1]
### Fixed
* Fixed a typo in the call to `imageMath.py`.
* `imageMath.py` is now called via `subprocess.run` rather than `os.system`.

## [0.8.0]
### Added
* Functions for resampling geographic image to radar coordinates, copying ISCE2 images, and performing ISCE2 image math to utils.py.
### Changed
* `create_water_mask` so that it pulls data from a partition parquet file (speeds up downloads), and added option to output water mask in any GDAL format.
* `insar_tops_burst` so that water masking is done pre-unwrapping if masking is requested.

## [0.7.2]
### Fixed
* Description of the range of the lv_phi in the readme.md.txt.j2.

## [0.7.1]
### Added
* insar_tops_burst now validates burst pair granule names.

## [0.7.0]
### Added
* InSAR stripmap workflow to support AVO's ALOS-1 processing efforts. This workflow is specific to AVO currently, and may not work for others.
### Changed
* The naming convention for the burst products has been updated.

## [0.6.2]
### Changed
* The geocoding DEM is now resampled to ~20m in the case of 5x1 looks.

## [0.6.1]
### Added
* `apply_water_mask` optional argument to apply water mask in the wrapped and unwrapped phase geotiff files

### Changed
* updated product readme template to include references to the water masking layer and the option to apply it
* updated repo readme to include information on processing options

## [0.6.0]
### Changed
* Pixel sizes of output GeoTIFFs now better reflect their resolution: 80m for 20x4 looks, 40m for 10x2 looks, and 20m for 5x1 looks.

## [0.5.0]
### Changed
* `insar_tops_burst` workflow now supports burst products advertised in CMR production, rather than CMR UAT

## [0.4.1]
### Added
* Generate a README file to be included with burst products that describes the processing workflow.

## [0.4.0]
### Changed
* `insar_tops_burst` workflow now takes the burst product names rather than the SLC product names.
* Replace `--range-looks` and `--azimuth-looks` options with a single `--looks` option for `insar_tops` and `insar_tops_burst` workflows
* `insar_tops_burst` workflow now always uses the oldest granule as the reference.

### Fixed
* Incomplete DEM generation issue by switching to using `merged/dem.crop` as the source

## [0.3.0]
### Added
* `insar_tops` workflow for processing of full Sentinel-1 SLCs.
* Ability to pass Earthdata username and password as environment variables to workflows. This allows the credentials to be passed to the Docker container via the `-e` option.
* The `++omp-num-threads` parameter for control of the number of threads used when ISCE2 calls OpenMP functionality.
* Added to the `insar_tops_burst` workflow:
  * Generation of output geotiff products with sensible names in subfolder that is also added as a zip archive.
  * Generation of a product browse image based on the unwrapped phase geotiff.
  * Generation of a parameter file for the burst products
  * Generation of output satellite geometry products (azimuth angle and lookup angle) to enable further time series processing.

## [0.2.1]
### Added
* The `get_isce2_burst_bbox` function to calculate burst bounding boxes using ISCE2 directly to fix inaccurate generation of burst bounding boxes.

### Removed
* The `reformat_gcp`, `create_gcp_df`, and `create_geometry` methods from the `BurstMetadata` class because they are superseded by `get_isce2_burst_bbox`

### Fixed
* `test_get_region_of_interest` so that it has an ascending and descending test case, not two descending. This required changing the `reference_ascending.xml` and `secondary_ascending.xml` files in `tests/data`.
* Fixed the path to the `entrypoint.sh` script in the Docker container image.

## [0.2.0]
### Added
* It's now possible to register multiple HyP3 entry points (workflows) and run them through the main hyp3_isce2 entry point. 
* A new entrypoint and skeleton process file for a stripmap process.
* Concurrent download functionality for burst extractor calls.

### Changed
* Burst workflow renamed to `insar_tops_burst` from `topsapp_burst`.

### Fixed
* A zero-index bug in `burst.py` that led to incorrect geolocation of products.
* Typo in the `release-checklist-comment` workflow.

## [0.1.0]
### Added
* Initial release of the HyP3 ISCE2 plugin.
