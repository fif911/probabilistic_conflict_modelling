# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2024-07-28

### Added

- Initial Implementation of the Military Conflict Forecasting Model

## [1.0] - 2023-08-17

### Added

- Converted .ipynb files to .py using Jupytex.
- Support for 2023 data.
- 11 PCAs for V-DEM dataset in the final `cm_features_v2.5` and removed 40 V-DEM features.
- New benchmarks published by ViEWS.

### Changed

- Migrated to the most recent version of `cm_features` published by ViEWS.

## [1.1] - 2023-08-31

### Added

- Support for 2024 year.
- Bash scripts for running data preprocessing pipeline and syncing Jupyter notebooks with Python files.

### Changed

- Bugfixes for line plots logic to ensure correct conversion of values in case log transform flag is on.
- Structure of the NGBoost model to simplify handling for 2024
