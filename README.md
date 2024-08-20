
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8007397.svg)](https://doi.org/10.5281/zenodo.8007397) ![Coverage](images/coverage.svg)

# HyP3 ISCE2 Plugin

The HyP3-ISCE2 plugin provides a set of workflows to process SAR satellite data using the [InSAR Scientific Computing Environment 2](https://github.com/isce-framework/isce2) (ISCE2) software package. This plugin is part of the [Alaska Satellite Facility's](https://asf.alaska.edu) larger HyP3 (Hybrid Plugin Processing Pipeline) system, which is a batch processing pipeline designed for on-demand processing of SAR data.

## Usage
The HyP3-ISCE2 plugin provides a set of workflows (accessible directly in Python or via a CLI) that can be used to process SAR data using ISCE2. The workflows currently included in this plugin are:

- `insar_stripmap`: A workflow for creating ALOS-1 geocoded unwrapped interferogram using ISCE2's Stripmap processing workflow
- `insar_tops`: A workflow for creating full-SLC Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow
- `insar_tops_burst`: A workflow for creating single-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow
- `insar_tops_multi_burst`: A workflow for creating multi-burst Sentinel-1 geocoded unwrapped interferogram using ISCE2's TOPS processing workflow
---

To run a workflow, simply run `python -m hyp3_isce2 ++process [WORKFLOW_NAME] [WORKFLOW_ARGS]`. For example, to run the `insar_tops_burst` workflow:

```
python -m hyp3_isce2 ++process insar_tops_burst \
  S1_136231_IW2_20200604T022312_VV_7C85-BURST \
  S1_136231_IW2_20200616T022313_VV_5D11-BURST \
  --looks 20x4 \
  --apply-water-mask True
```

and, for multiple burst pairs:

```
python -m hyp3_isce2 ++process insar_tops_multi_burst \
  --reference S1_136231_IW2_20200604T022312_VV_7C85-BURST S1_136232_IW2_20200604T022315_VV_7C85-BURST \
  --secondary S1_136231_IW2_20200616T022313_VV_5D11-BURST S1_136232_IW2_20200616T022316_VV_5D11-BURST \
  --looks 20x4 \
  --apply-water-mask True
```

These commands will both create a Sentinel-1 interferogram that contains a deformation signal related to a 2020 Iranian earthquake.

### Product Merging Utility Usage
**This feature is under active development and is subject to change!**

Burst InSAR products created using the `insar_tops_burst` workflow can be merged together using the `merge_tops_burst` workflow. This can be useful when the deformation signal you'd like to observe spans multiple bursts. It can be called using the following syntax:
```
python -m hyp3_isce2 ++process merge_tops_bursts \
  PATH_TO_UNZIPPED_PRODUCTS \
  --apply-water-mask True
```
Where `PATH_TO_UNZIPPED_PRODUCTS` is the path to a directory containing unzipped burst InSAR products. For example:
```bash
PATH_TO_UNZIPPED_PRODUCTS/
├─ S1_136232_IW2_20200604_20200616_VV_INT80_663F/
├─ S1_136231_IW2_20200604_20200616_VV_INT80_529D/
```
In order to be merging eligible, all burst products must:
1. Have the same reference and secondary dates
1. Have the same polarization
1. Have the same multilooking
1. Be from the same relative orbit
1. Be contiguous

The workflow should through an error if any of these conditions are not met.

**Merging burst InSAR products requires extra data that is not contained in the production HyP3 Burst InSAR products. For the time being, to be merging eligible burst products must be created locally using your own installation of `hyp3-isce2` from the `merge_bursts` branch of this repository!**

As mentioned above this feature is under active development, so we welcome any feedback you have!

### Options
To learn about the arguments for each workflow, look at the help documentation 
(`python -m hyp3_isce2 ++process [WORKFLOW_NAME] --help`).

#### Looks Option
When ordering Sentinel-1 Burst InSAR On Demand products, users can choose the number of **looks** (`--looks`) to use 
in processing, which drives the resolution and pixel spacing of the output products. The available options are 
20x4, 10x2, or 5x1. The first number indicates the number of looks in range, the second is the number of looks 
in azimuth.

The output product pixel spacing depends on the number of looks in azimuth:
pixel spacing = 20 * azimuth looks

Products with 20x4 looks have a pixel spacing of 80 m, those with 10x2 looks have a pixel spacing of 40 m, and
those with 5x1 looks have a pixel spacing of 20 m.

#### Water Mask Option
There is always a water mask geotiff file included in the product package, but setting the **apply-water-mask** 
(`--apply-water-mask`) option to True will apply the mask to the wrapped interferogram prior to phase unwrapping.

### Earthdata Login Credentials

For all workflows, the user must provide their Earthdata Login credentials in order to download input data.
If you do not already have an Earthdata account, you can sign up [here](https://urs.earthdata.nasa.gov/home). 
Your credentials can be passed to the workflows via environment variables
(`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`) or via your `.netrc` file. If you haven't set up a `.netrc` file
before, check out this [guide](https://harmony.earthdata.nasa.gov/docs#getting-started) to get started.

### Docker Container
The ultimate goal of this project is to create a docker container that can run ISCE2 workflows within a HyP3 
deployment. To run the current version of the project's container, use this command:
```
docker run -it --rm \
    -e EARTHDATA_USERNAME=[YOUR_USERNAME_HERE] \
    -e EARTHDATA_PASSWORD=[YOUR_PASSWORD_HERE] \
    ghcr.io/asfhyp3/hyp3-isce2:latest \
    ++process [WORKFLOW_NAME] \
    [WORKFLOW_ARGS]
```

**NOTE** Each workflow can also be accessed via an alternative CLI with the format (`[WORKFLOW_NAME] [WORKFLOW_ARGS]`)

#### Docker Outputs

To retain hyp3_isce2 output files running via Docker there are two recommended approaches:

1. Use a volume mount

Add the `-w /tmp -v [localdir]:/tmp` flags after docker run. `-w` changes the working directory of the container to `/tmp` and `-v` will mount whichever local directory you choose so that such that hyp3_isce3 outputs are preserved locally.

1. Copy outputs to remote object storage

Append the `--bucket` and `--bucket-prefix` to [WORKFLOW_ARGS]. *Only the final output files and zipped archive of those files is uploaded.* This also requires that AWS credentials to write to the bucket are available to the running container. For example, to write outputs to a hypothetical bucket `s3://hypothetical-bucket/test-run/`:

```
docker run -it --rm \
    -e AWS_ACCESS_KEY_ID=[YOUR_KEY] \
    -e AWS_SECRET_ACCESS_KEY=[YOUR_SECRET] \ 
    -e AWS_SESSION_TOKEN=[YOUR_TOKEN] \ 
    -e EARTHDATA_USERNAME=[YOUR_USERNAME_HERE] \
    -e EARTHDATA_PASSWORD=[YOUR_PASSWORD_HERE] \
    ghcr.io/asfhyp3/hyp3-isce2:latest \
      ++process [WORKFLOW_NAME] \
      [WORKFLOW_ARGS] \
      --bucket "hypothetical-bucket" \
      --bucket-prefix "test-run"
```

Tip: you can use [`docker run --env-file`](https://docs.docker.com/reference/cli/docker/container/run/#env) to capture all the necessary environment variables in a single file.


## Developer Setup
1. Ensure that conda is installed on your system (we recommend using [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to reduce setup times).
2. Download a local version of the `hyp3-isce2` repository (`git clone https://github.com/ASFHyP3/hyp3-isce2.git`)
3. In the base directory for this project call `mamba env create -f environment.yml` to create your Python environment, then activate it (`mamba activate hyp3-isce2`)
4. Finally, install a development version of the package (`python -m pip install -e .`)

To run all commands in sequence use:
```bash
git clone https://github.com/ASFHyP3/hyp3-isce2.git
cd hyp3-isce2
mamba env create -f environment.yml
mamba activate hyp3-isce2
python -m pip install -e .
```

## Background
HyP3 is broken into two components: the cloud architecture/API that manage processing of HyP3 workflows, and Docker container plugins that contain scientific workflows which produce new science products from a variety of data sources (see figure below for the full HyP3 architecture).

![Cloud Architecture](images/arch_here.jpg)

The cloud infratstructure-as-code for HyP3 can be found in the main [HyP3 repository](https://github.com/asfhyp3/hyp3). This repository contains a plugin that can be used to process ISCE2-based processing of SAR data.

This project was heavily influenced by the [DockerizedTopsApp](https://github.com/ACCESS-Cloud-Based-InSAR/DockerizedTopsApp) project, which contains a similar workflow that is designed to produce ARIA Sentinel-1 Geocoded Unwrapped Interferogram standard products via HyP3.

## License
The HyP3-ISCE2 plugin is licensed under the Apache License, Version 2 license. See the LICENSE file for more details.

## Code of conduct
We strive to create a welcoming and inclusive community for all contributors to HyP3-ISCE2. As such, all contributors to this project are expected to adhere to our code of conduct.

Please see `CODE_OF_CONDUCT.md` for the full code of conduct text.

## Contributing
Contributions to the HyP3-ISCE2 plugin are welcome! If you would like to contribute, please submit a pull request on the GitHub repository.

## Contact Us
Want to talk about HyP3-ISCE2? We would love to hear from you!

Found a bug? Want to request a feature?
[open an issue](https://github.com/ASFHyP3/asf_tools/issues/new)

General questions? Suggestions? Or just want to talk to the team?
[chat with us on gitter](https://gitter.im/ASFHyP3/community)
