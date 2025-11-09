# Parallel-EM-Clustering

Implementation of **Expectation–Maximization** (EM) algorithm for clustering, using **MPI** for distributed processing.

## Repository Contents

---

## Program Execution

### 1. Compilation

To build the project, first load the **MPICH** module and compile:

```bash
module load mpich-3.2
make
```

The executable will be generated at `bin/EM_Clustering`.

---

### 2. Generating the Input Data

Before running the Python script on your personal machine, a virtual environment should be created with all the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib
```

Then, from the main directory the script can be launched with:

```bash
python data/datasetGeneration/data-generator.py
```

#### Parameters

The list of possible parameters for the script are:

- `--samples, -s` (**int**, default `1000`):  
  Number of data points (samples) to generate.

- `--features, -f` (**int**, default `2`):  
  Number of features (dimensions) for each data point.

- `--clusters, -k` (**int**, default `3`):  
  Number of clusters (Gaussian components) to generate.

- `--means` (**list of str**):  
  One mean vector per cluster.  
  Example:  
  ```bash
  --means -5,0 0,5 5,0
  ```
  Each cluster mean should have a number of elements equal to `--features`.

- `--std` (**list of float**, default `1.0`):  
  Standard deviation(s) for the clusters. You can specify:
  - A single value to use for all clusters, or  
  - One value per cluster (e.g. `--std 1.0 0.8 1.2`).

- `--equal_size, -equal` (**flag**, default `False`):  
  Generate clusters with equal sample sizes instead of random proportions.

- `--random_state, -r` (**int**, default `43`):  
  Random number generator seed for reproducibility.

- `--output, -o` (**str**, default `em_dataset.csv`):  
  Output CSV filename. Saved under the `data/` directory.

- `--metadata, -m` (**str**, default `em_metadata.txt`):  
  Metadata filename (also saved under the `data/` directory).

- `--plot` (**flag**):  
  Display a 2D scatter plot of the generated dataset (only works when `--features 2`).

---

#### Import the Dataset to the Cluster

The generated files can be exported to the cluster via:

```bash
scp data/em_dataset.csv data/em_metadata.txt user@cluster:/path/to/destination

```

---

### 3. Running the Program

Once the dataset is ready and imported to the cluster, run the program with MPI.

**Single run:**

```bash

```

**Batch/multiple runs (for performance experiments):**

We provide a launcher that generates job scripts for different node/core combinations and submits them to the cluster:

```bash
./scripts/multiple_job_launcher.sh
```

This script:
- Generates job scripts from `scripts/job_template.sh` by substituting node/core, placement, and parameters.
- Submits them via `qsub` to the job queue of the cluster.

---

## Outputs

---

## License & Authors
