# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.4.0]
### Changed
* `insar_tops_burst` workflow now takes the burst product names rather than the SLC product names.

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
