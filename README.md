# DPNet: Addressing Cross-Segment Heterogeneity in Multiscale Upsampling for Long-Term Time Series Forecasting

All scripts, models, and data instructions are structured for easy reproduction and extension of experiments.

---

## 1. Environment Setup

We recommend using Anaconda/Miniconda to manage the Python environment.

### Create and Activate Environment

```bash
conda create -n dpnet_env python=3.8 -y
conda activate dpnet_env

# in the repository root
pip install -r requirements.txt
```

---

## 2. Datasets

We provide two example datasets for quick testing:

- `./dataset/exchange_rate.csv`
- `./dataset/weather.csv`

The full required datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

Place (or verify) the files under the `dataset/` directory:

```
dataset/
├── exchange_rate.csv
└── weather.csv
```

---

## 3. Project Structure

The repository is organized as follows for quick verification and extension:

```
.
├── dataset/                # Example CSV datasets for quick testing
│   ├── exchange_rate.csv
│   └── weather.csv
├── models/
│   └── DPNet.py            # Main proposed model implementation (DPM + PRR)
├── scripts/
│   ├── exchange_rate.sh    # Run DPNet on Exchange-Rate
│   └── weather.sh          # Run DPNet on Weather
├── run_longExp.py          # Main entry to run experiments
├── requirements.txt
└── README.md
```
---

## 4. Running Main Experiments

We temporarily provide the following two scripts for testing and verification:

```bash
bash ./scripts/exchange_rate.sh
bash ./scripts/weather.sh
```

### Results

The prediction results of each script are saved in:

```
./result.txt
```

### Permission Error

If you encounter `sh: Permission denied`, run:

```bash
find ./scripts -type f -name "*.sh" -exec chmod +x {} \;
```
