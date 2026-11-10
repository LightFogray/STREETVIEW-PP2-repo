# Replication package: value of a reference image's voting history for street-safety prediction

This repository accompanies a manuscript under double-anonymous review. It contains the derived
tables, fold definitions, evaluation scripts and aggregated results needed to reproduce every
number in the paper. No author-identifying information is included, by design.

## What the study does

Given two street-level images, a visual classifier predicts whether survey participants chose the
left image, the right image, or a tie. The paper asks how much a *reference image's* existing
voting history adds on top of what the images themselves show, when the whole city is held out of
training.

## Data

The analyses use the Place Pulse 2.0 archive as distributed by the Centre for Collective Learning:

```
URL:    https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip
bytes:  2,640,932,754
sha256: see data/ARCHIVE_CHECKSUMS.txt
```

Files used from the archive: `votes.tsv`, `studies.tsv`, `locations.tsv`, `places.tsv`, and
`images/*.JPG` (110,988 images). Street-level photographs are **not** redistributed here, because
they remain subject to the imagery provider's terms. Download the archive and unpack it to
`data/raw/placepulse2/` before running the scripts.

Two derived files are included:

| File | Size | Description |
|---|---|---|
| `data/pairs.parquet` | 29 MB | 506,165 valid safety comparisons with image IDs, city labels, coordinates, choice, participant identifier and timestamp |
| `data/logo_folds.json` | 12 KB | The 15 leave-one-city-out folds, with test and validation city per fold |
| `data/image_ids.csv` | 3.6 MB | The 111,389 image identifiers referenced by the votes, with city labels |

`data/ARCHIVE_CHECKSUMS.txt` records the archive size and SHA-256.

## Repository structure

```
data/       fold definitions, derived vote table, image identifiers, archive checksums
scripts/    the analysis pipeline, in execution order
results/    aggregated per-fold outputs behind Tables 2 and 3 and the inference section
figures/    the four manuscript figures, as vector PDF and 400 dpi PNG
manuscript/ the submitted main document, title page and supplementary material
```

## Scripts and what they produce

Run in this order. Each script writes its output to `outputs/` (not included here except as
aggregated files in `results/`).

| Step | Script | Produces |
|---:|---|---|
| 1 | `build_tables.py` | `pairs.parquet`; prints the archive audit reported in Section 2 |
| 2 | `make_splits.py` | `logo_folds.json`; runs the leakage assertions on all 15 folds |
| 3 | `verify_pp2_images.py` | image availability, coordinate integrity, the location checklist |
| 4 | `extract_features.py` | 768-dimensional frozen ConvNeXt-Tiny features (110,987 x 768, 151 MB, about 12 minutes on an RTX 4060) |
| 5 | `converge_heads.py` | Table 3: history gain by comparison head and training budget |
| 6 | `anchor_history_fusion.py` | Table 2: reference history, matched shuffle, coefficient selection |
| 7 | `bt_history.py` | the opponent-adjusted (Davidson) history summary, three seeds |
| 8 | `inference_checks.py` | the city-cluster bootstrap interval and the permutation null |
| 9 | `make_figures.py` | Figures 1 to 4 |
| 10 | `audit_all_numbers.py` | re-derives and checks every number in the manuscript |

`shared_scorer_check.py` is the additional comparison head reported in Section 3.

## Reproducing the headline numbers

```
Table 2, row C        0.905491 -> 0.904930   (-0.000561, 0.062% relative)
Table 2, row C'       0.905036              (-0.000455, 0.0502%)
Table 2, shuffle      1,000 draws, mean +0.000206, central 95% +0.000159 to +0.000254
Table 2, reselected   200 draws, mean +0.000106
City bootstrap        95% interval -0.000792 to -0.000345 (0.038% to 0.087% relative)
Record-weighted       -0.000543
Table 3               difference head 0.055% (60 ep); expanded head 0.055-0.081%;
                      shared scorer 5.83% (4 ep) and 1.26% (150 ep)
```

`results/inference_summary.json` holds the bootstrap and permutation values; the per-fold files in
`results/` hold the inputs.

## Paths

The scripts read from three absolute paths defined near the top of each file:

```python
DATA = Path(r".../STREETVIEW-PP2/data")        # derived tables and features
OUT  = Path(r".../STREETVIEW-PP2/outputs")     # script outputs
RAW  = Path(r".../data/raw/placepulse2")       # the unpacked archive
```

Set these to your own locations before running. `extract_features.py` also expects the archive's
`place-pulse-2.0.zip` in `RAW`.

## Environment

Python 3.11 or later. See `requirements.txt`. A CUDA device is used when available; the supervised
heads train in minutes on a single laptop GPU, and `extract_features.py` is the only long step.

## Scope and known limits

- The 15 test cities are those with at least 300 within-city comparisons before feature matching.
  They describe this archive, not a random sample of cities.
- Each comparison carries a single response, so disagreement on a given pair cannot be estimated
  directly.
- The history treatment is one shrunken net-win score with a fixed shrinkage constant. A richer
  summary is a different treatment.
- The 1.5% figure quoted in Section 3 is a target this project set, not an established standard.

## License

Code released under the MIT licence. Derived tables are released under CC BY 4.0. The underlying
imagery remains subject to the provider's terms and is not redistributed.
