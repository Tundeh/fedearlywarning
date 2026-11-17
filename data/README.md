# Data

The raw datasets are **not redistributed** in this repository. Download them
from the Canadian Institute for Cybersecurity (CIC) and accept their terms of
use, then preprocess them with the scripts below.

## Expected layout

```
data/
├── CSE-CIC-IDS2018/                 # raw (10 CSVs, ~6.7 GB)
│   ├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
│   ├── ...                          # all 10 "*_TrafficForML_CICFlowMeter.csv" files
│   └── Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv   # (sic) official file name
├── CICIoT2023/                      # raw (309 CSVs in 34 per-class folders)
│   ├── Benign_Final/BenignTraffic*.pcap.csv
│   ├── Mirai-udpplain/Mirai-udpplain*.pcap.csv
│   └── ...
└── processed/                       # created by the prepare scripts
    ├── cicids2018.npz  cicids2018.json
    └── ciciot2023.npz  ciciot2023.json
```

## CSE-CIC-IDS2018

- Source: <https://www.unb.ca/cic/datasets/ids-2018.html> (AWS Open Data
  bucket `s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/`).
- Use the ten "Processed Traffic Data for ML Algorithms" CSV files
  (`*_TrafficForML_CICFlowMeter.csv`). All rows are used.
- Prepare: `bash scripts/prepare_cse.sh`
  (`utils/prepare_cicids2018.py`, 20 clients auto-detected from `Dst IP`, no row cap,
  20% scaler sample excluding the botnet family, seed 0).
- Result: 16,233,002 rows, d = 70 features, C = 8 classes.

## CICIoT2023

- Source: <https://www.unb.ca/cic/datasets/iotdataset-2023.html>
  ("CSV" per-class folders, one folder per attack/benign class).
- The class label is taken from the **folder / file name**; files without an
  internal label column are kept, not skipped.
- Prepare: `bash scripts/prepare_ciciot.sh`
  (`utils/prepare_ciciot2023.py`, 20 deterministic logical clients from
  chronological file blocks, first 50,000 rows of each CSV,
  20% scaler sample excluding the botnet family, seed 0).
- Result: 12,572,776 rows from 309 CSV files, d = 39 features, C = 7 classes.
- The CICIoT2023 stream ordering and client assignment are a **deterministic
  experimental construction**, not a verified common real-world chronology.

## Verifying preprocessing

The SHA-256 checksums of the processed files used for every paper run are in
[`../provenance/processed_data_sha256.txt`](../provenance/processed_data_sha256.txt):

```
f6b69380acf29d55890059839b1af0083fa825e78f3d898509698e79c7205a9b  data/processed/cicids2018.npz
68ee57091a85c7fa4098856774fd0a0ed934831f70fbcba5144acab1a425a2d8  data/processed/ciciot2023.npz
```

Both prepare scripts print the expected and the obtained checksum. Small
schema summaries (class, variant and feature names, row counts, source file
list) are in [`../data_schema/`](../data_schema/).
