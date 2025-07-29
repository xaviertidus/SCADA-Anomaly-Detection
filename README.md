# SCADA Anomaly Detection Analysis

Replicates anomaly detection from Shirazi et al. (2016) for SCADA systems using K-means, NB, PCA-SVD, GMM, and Isolation Forest. Includes Python scripts and Jupyter Notebook for preprocessing and evaluation. Credit repo and authors if used.

## Source Material
- Evaluation of Anomaly Detection techniques for SCADA communication resilience https://ieeexplore-ieee-org.wwwproxy1.library.unsw.edu.au/document/7573322
- Datasets: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

## Prerequisites

- macOS (tested on recent versions; should work on Linux/Windows with minor adjustments)
- Python 3.8+ (install via Homebrew on Mac: `brew install python`)
- Git (pre-installed on Mac or via `brew install git`)

## Setup

1. **Clone the Repository**:
   ```
   git clone https://github.com/xaviertidus/SCADA-Anomaly-Detection
   cd SCADA-Anomaly-Detection
   ```

2. **Create and Activate Virtual Environment**:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```
   pip install --upgrade pip
   pip install pandas numpy scipy scikit-learn matplotlib
   ```
   For Jupyter: `pip install jupyter`.

4. **Prepare the Dataset**:
   - Download the full dataset (`Dataset4.arff` or `Dataset4.txt`) from the source (e.g., Mississippi State University SCADA Lab or provided link).
   - Place the file(s) in the root of the repository.
   - Preferred: Use `Dataset4.arff` for direct loading. If using TXT, ensure it's comma-separated.

## Running the Analysis

### Option 1: Via Script (Batch Mode)
- Run:
  ```
  python analysis.py
  ```
- Outputs: Console table with metrics (Precision, Recall, etc.).

### Option 2: Via Jupyter Notebook (Interactive Mode)
- Start Jupyter: `jupyter notebook`.
- Open `analysis.ipynb`.
- Run cells sequentially (Shift+Enter).
- Outputs: Interactive results, including DataFrame tables and optional plots.

Both approaches perform the same steps: data loading/preprocessing, model training, and evaluation. They include a notice to credit the repository and the paper authors: Shirazi et al. (2016). "Evaluation of Anomaly Detection Techniques for SCADA Communication Resilience".

## Troubleshooting

- **ARFF Loading Issues**: If `scipy.io.arff` fails, fallback to CSV reading is automatic.
- **Memory Errors**: For large datasets, run on a machine with >8GB RAM or process in chunks (modify code with `pd.read_csv(chunksize=...)`).
- **Model Warnings**: Scikit-learn may warn about convergence; adjust hyperparameters (e.g., `n_init` for K-means).
- **Deactivate Env**: Run `deactivate` when done.

## Extensions

- For multi-class (categorized anomalies): Modify labels to use 'categorized result' and adjust metrics (e.g., `average='macro'`).
- Add visualizations: Uncomment plotting code if added.
- Contribute: Fork and PR improvements!

## License

MIT License. See LICENSE file for details.

## References

- Paper: [Shirazi et al. (2016). Evaluation of Anomaly Detection Techniques for SCADA Communication Resilience](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets){:target="_blank"}
- Dataset: [Mississippi State University SCADA Gas Pipeline Dataset](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets){:target="_blank"}