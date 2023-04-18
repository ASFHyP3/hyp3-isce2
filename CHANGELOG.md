# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]
### Added
* It's now possible to register multiple [HyP3 entry points (workflows) and run them through the main hyp3_isce2 entry point. 

### Changed
* Burst workflow renamed to `insar_tops_burst` from `topsapp_burst`

## [0.1.3]
### Fixed
* A zero-index bug in `burst.py` that led to incorrect geolocation of products

## [0.1.2]
### Added
* Concurrent download functionality for burst extractor calls.

## [0.1.1]
### Fixed
* Typo in the `release-checklist-comment` workflow.

## [0.1.0]
### Added
* Initial release of the HyP3 ISCE2 plugin.
