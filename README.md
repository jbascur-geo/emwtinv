# EMWTINV

**EMWTINV** is a Python tool for performing **Hybrid Bayesian Inversion (HBI)** for hydrogeophysical data interpretation.

## What is Hybrid Bayesian Inversion (HBI)?

Hybrid Bayesian Inversion (HBI) is a Bayesian extension of Coupled Hydrogeophysical Inversion (CHI) that enables efficient estimation of hydrological properties from geophysical data. It integrates:

* Petrophysical relationships
* Hydrological models
* Uncertainty in calibration parameters

HBI decomposes the subsurface model into two components:

* **Groundwater Model**:

  * Predictable using petrophysical and hydrological modeling
  * Characterized by **posterior probability density functions (PDFs)**

* **Background Model**:

  * Not predictable from petrophysical or hydrological models
  * Represented by deterministic resistivity structures (1D/2D/3D)
  * Not explicitly associated with uncertainty

This decomposition allows computationally expensive Bayesian inference to be applied only to the groundwater component (which is lower-dimensional), while the background model is handled using conventional least-squares inversion.

### Expectation-Maximization Algorithm

To jointly estimate:

* The posterior PDFs of the **Groundwater Model**
* The maximum likelihood estimate (MLE) of the **Background Model**

HBI uses the **Expectation-Maximization (EM)** algorithm:

* **E-step**: Bayesian inversion (via Metropolis-Hastings sampling)
* **M-step**: Least-squares inversion of the background model

---

## How It Works

EMWTINV is executed via the `emwtinv_multi.py` script. Example applications are located in the `/Script` directory.

### Directory Structure

* `/Source`: Source code
* `/Script`: Example scripts for applying EMWTINV
* `/Utils`: Tools for plotting and analyzing results

### Configuration Files

Before running, place the following files in the same directory as `emwtinv_multi.py`:

#### `emwtinv_setup.txt`

Defines input data, geophysical method, E-step, and M-step parameters.

**Content:**

**Geophysical Data**

```
Data_list: xxxx.dat        # Observation file with geophysical data
Method: DC2D               # Geophysical method
                           # Options:
                           # DC2D: Electrical Resistivity Tomography
                           # DC1D: Electrical Vertical Sounding
                           # TEM1D: Electromagnetic transient
                           # MT1D: Magnetotellurics (non-tested)
                           # GRAV2D: Gravity (non-tested)
```

**M-STEP Parameters (Background model inversion)**

```
externalmodel: True
external_constant_model: False
externalmodel_ref: True

model_file: path/to/model_file.txt
model_mesh: path/to/mesh_file.txt
modelref_file: path/to/modelref.txt
modelref_mesh: path/to/meshref.txt
invert_initial_bkmodel: True
reference_value: 400
sub_lmd:
emmax_iter: 100
```

**SimPEG Regularization Parameters (initial inversion)**

```
invparameters0.Collingrate: 5
invparameters0.Beta: 1.0
invparameters0.Minbeta: 1e-5
invparameters0.Collingfactor: 0.8
invparameters0.alpha_x: 1.0
invparameters0.alpha_y: 1.0
invparameters0.alpha_z: 1.0
invparameters0.alpha_s: 1.0
invparameters0.chifact: 1.0
```

**SimPEG Regularization Parameters (background model iterations)**

```
invparameters.Collingrate: 5
invparameters.Beta: 1.0
invparameters.Minbeta: 1e-5
invparameters.Collingfactor: 0.8
invparameters.alpha_x: 1.0
invparameters.alpha_y: 1.0
invparameters.alpha_z: 1.0
invparameters.alpha_s: 1.0
invparameters.chifact: 1.0
```

**E-STEP Parameters (Bayesian inversion)**

```
estep_niter: 1000
estep_beta:
estep_beta_factor:
estep_prop: 2D         # Options: REF, 2D, 1D
vmin:
vmax:
```

#### `sigmawt_setup.txt`

Defines prior PDFs and parameter ranges for the groundwater model.

**Example (2D unconfined aquifer - ERT):**

```
Zmin: 10
Zmax: 30
dZ: 0.5
log_sigma_min: -1
log_sigma_max: 9
beta_min: -1
beta_max: -0.05
por_min: 0.10
por_max: 0.50
m1: 1.5
m2: 2.5
logA1: 0.0
logA2: 3.4
sigma0:
sigma0_var:
magic_lmd:
```

---

## Running EMWTINV

### Single-process Mode

```bash
python emwtinv_multi.py
```

### Parallel Mode (Linux only)

```bash
mpirun -n <number_of_processes> python -m emwtinv_multi.py
```

> **Note:** Parallel computing is used **only during the E-step** for Bayesian sampling. The total number of samples will be:
>
> `number_of_processes × samples_per_process`

Parallelization is implemented via the `mpi4py` library. On Windows, `mpi4py` supports only single-process execution.

---

## Dependencies

* **[SimPEG](https://simpeg.xyz/)**: Geophysical simulation and inversion

  ```bash
  pip install simpeg
  ```

* **[mpi4py](https://mpi4py.readthedocs.io/)**: MPI for parallel Bayesian sampling

  ```bash
  pip install mpi4py
  ```

  > ⚠️ Full parallel execution is supported only on Linux

* **NumPy / SciPy**: Core scientific libraries

  ```bash
  pip install numpy scipy
  ```

---

## Installing EMWTINV

1. Clone the repository

```bash
git clone https://github.com/your-username/emwtinv.git
cd emwtinv
```

2. Install the required Python libraries (see above)

3. Explore the following directories:

   * `/Source`: Core source code
   * `/Script`: Example applications
   * `/Utils`: Visualization and result analysis tools

---

## License

MIT License
