# REIMS with four energy-based devices for detection of cutaneous SCC

Data and analysis code for

> Brorsen LF, Granborg JR, Janfelt C, Haedersdal M, Bjerring P, Lerche CM.
> *Comparison of Energy-Based Devices for Mass Spectrometric Detection of Skin Cancer During Tissue Removal.*
> Lasers in Surgery and Medicine (submitted).

The notebook reproduces the analysis of the paper from the peak-picked spectral matrix: outlier
detection, mouse-level data partitioning, feature selection with nested cross-validation, the
device-independent model, the intra- and cross-device models, and the supplementary tables and figures.

## Contents

| Path | Description |
|---|---|
| `reims_multidevice_analysis.ipynb` | The analysis notebook |
| `reims_multidevice_analysis.py` | The same code as a plain Python script |
| `data/reims_peakpicked_spectra.npz` | `X`: 933 spectra x 10,480 consensus peaks (raw ion counts); `mz`: peak positions; `X_burn`: burn-level matrix (100 x 10,480) |
| `data/reims_spectrum_metadata.csv` | One row per spectrum: burn file, scan index, tissue, device, sample ID |
| `data/reims_burn_metadata.csv` | One row per burn/acquisition |
| `output/` | Tables and figures produced by the notebook (included for reference) |

## Dataset

* 12 hairless mice with UV-induced squamous cell carcinoma, sampled *post mortem*.
* Four energy-based devices: super-pulsed CO2 laser (`SP_CO2`), pulsed CO2 laser (`P_CO2`),
  electrocautery (`Cautery`) and nitrogen plasma (`N2_plasma`).
* 95 burns (one data file per continuous burn), 877 tumor and non-tumor spectra (1 s scans, negative ion mode).
* 56 background spectra (surrounding air) are included in the matrix.
* `sample_id` is the mouse number for skin spectra. All spectra of a mouse are kept in the same set
  (training: 8 mice; hold-out: 4 mice).

## Upstream processing (not repeated in the notebook)

Performed in MS IonVision:

1. Conversion of Waters .raw files to centroided mzML (ProteoWizard msconvert).
2. Selection of the scans belonging to each burn; ions within *m/z* 200-1,400 with intensity >= 10 counts.
3. Consensus peak picking on class-average spectra (local maximum, window 11 points, minimum height 0.1 % of
   the class maximum) and extraction of peak intensities per spectrum (tolerance 0.01 Da). The consensus peak list
   was generated in a processing run that also included spectra of murine liver, which are not part of this
   dataset; peaks not detected in skin or background spectra are removed by the prevalence filter.

## Running

```bash
pip install -r requirements.txt
jupyter notebook reims_multidevice_analysis.ipynb      # or: python reims_multidevice_analysis.py
```

Run time is about three minutes on a laptop. All random steps use a fixed seed (42).

## Key results reproduced

| Result | Value |
|---|---|
| Outliers removed | 73 of 933 spectra |
| Training / hold-out mice | 8 / 4 |
| Features after 30 % prevalence filter (three classes) | 1,135 |
| Selected feature panel | 170 features |
| Device-independent model, >= 80 % confidence | accuracy 99.3 % (273/275), coverage 96.5 % (275/285) |
| Device-independent model, all spectra | accuracy 98.2 % (280/285) |
| Cross-device matrix | `output/TableSM6_cross_device.csv` |

## Licence and citation

[Add a licence, e.g. MIT for the code and CC BY 4.0 for the data, and the final citation and DOI.]
