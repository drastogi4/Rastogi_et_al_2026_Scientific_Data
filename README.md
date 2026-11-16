# Rastogi_et_al_2026_Scientific_Data

# Storm-Type Identification and Precipitation Attribution

This repository contains scripts used to identify storm types in ERA5, attribute precipitation to storm classes, and calculate storm-type likelihoods for the associated manuscript. It supports three analysis configurations:

- **TE** — TempestExtremes identification
- **TRACK–TE** — TRACK identification of hurricanes and extratropical cyclones (ETCs), combined with TE identification of atmospheric rivers (ARs) and mesoscale convective systems (MCSs)
- **WNet** — storm identification using WNet products

The scripts are organized by workflow stage. They process external input datasets and write derived products; the large input and output data files are not included in this repository.

## Repository contents

### Storm identification

| Script | Purpose |
|---|---|
| `identify_AR_TE.csh` | Identify ARs with TE and remove AR detections within 8° of TE hurricane tracks. |
| `identify_ERC_TE.csh` | Identify and stitch ETC tracks with TE. |
| `identify_HURRICANE_TE.csh` | Identify hurricanes with TE; writes CSV and GFDL-format tracks. |
| `identify_MCS_TE.csh` | Identify MCSs with TE using OLR, cold-core, precipitation, area, and lifetime criteria. |

TRACK and WNet storm-identification scripts will be added separately. The TE shell scripts use C-shell syntax and have the `.csh` extension.

### Precipitation masks

| Script | Purpose |
|---|---|
| `extract_ETC_precipitation_mask_from_TRACK.py` | Extract a precipitation mask using TRACK ETC tracks. |
| `extract_ETC_precipitation_mask_TE.py` | Extract a precipitation mask using TE ETC tracks. |
| `extract_ETC_precipitation_mask_WNet.py` | Extract a precipitation mask using WNet ETC products. |
| `extract_HURRICANE_precipitation_mask_TE.py` | Extract a precipitation mask using TE hurricane tracks. |
| `extract_HURRICANE_precipitation_mask_TRACK.py` | Extract a precipitation mask using TRACK hurricane tracks. |
| `extract_HURRICANE_precipitation_mask_WNet.py` | Extract a precipitation mask using WNet hurricane products. |

### Storm-type classification

These NCL scripts apply the common precipitation-classification workflow to the three method configurations:

- `stormtyping_TE_6hr.ncl`
- `stormtyping_TRACK-TE_6hr.ncl`
- `stormtyping_Wnet_6hr.ncl`

Storm-type codes used by the classification are:

| Code | Class |
|---:|---|
| 0 | Unclassified / no assigned storm type |
| 1 | MCS |
| 2 | ETC |
| 3 | Hurricane |
| 4 | AR |
| 5 | Other convective precipitation |

Other convective precipitation is assigned where ERA5 convective precipitation is positive and none of the storm masks is present. For TRACK–TE, TRACK identifies hurricanes and ETCs; TE identifies ARs and MCSs. Document any method-specific overlap precedence in the corresponding script header and manuscript methods.

### Likelihood and plotting

- `stormtyping_6hr_likelihood.py` — calculate six-hourly storm-type likelihood products.
- `plot_likelihood_TE.py` — plot TE likelihood results.
- `plot_stormtype.py` — plot storm-type products.

## Data and output conventions

The scripts expect input files to be available at paths configured in each script. Paths in the current TE scripts are examples based on the original analysis directory layout; update them for your system before running. Expected input products include annual ERA5 precipitation, convective precipitation, IVT, OLR/rain, storm-track, and storm-mask files, as applicable to each workflow.

Generated NetCDF files, track files, temporary files, and plots should be written outside version control. The `.gitignore` excludes common generated data and intermediate files. Keep configuration and scripts in Git; use the manuscript's data availability statement or an archival repository for datasets and final data products.

## Software requirements

Requirements vary by workflow. The TE identification scripts use:

- TempestExtremes executables, including `DetectNodes`, `StitchNodes`, `DetectBlobs`, `StitchBlobs`, and `NodeFileFilter` as needed
- Slurm (`sbatch`/`srun`) on the configured HPC system
- NetCDF libraries
- CDO for the MCS precipitation-filtering workflow

The classification scripts require NCL and its standard contributed script libraries. The Python scripts require Python and the packages they import (document exact package versions in an environment file as the workflows are finalized).

## Running the TE identification scripts

From the repository root, submit the relevant C-shell script to Slurm after checking its input paths and resource settings:

```bash
sbatch identify_AR_TE.csh
sbatch identify_ETC_TE.csh
sbatch identify_HURRICANE_TE.csh
sbatch identify_MCS_TE.csh
```

The scripts currently default to these year ranges:

- AR: 1980–2024
- ETC: 1980–2024
- Hurricane: 2000–2004 (change `start_year` and `end_year` in the script for the full analysis period, if intended)
- MCS: 1980–2024

Check each script's header and configuration section for input filenames, output names, thresholds, and cluster settings. Run a short test year first, then verify the generated files before submitting the full period.

