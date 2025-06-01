
![emwtinv_logo](https://github.com/user-attachments/assets/d1dc7074-3ec4-4dee-a239-82fe64a388b3)

**EMWTINV** is a Python tool for performing **Hybrid Bayesian Inversion (HBI)** for hydrogeophysical data interpretation.

## What is Hybrid Bayesian Inversion (HBI)?

Hybrid Bayesian Inversion (HBI) is a Bayesian extension of Coupled Hydrogeophysical Inversion (CHI) that enables efficient estimation of hydrological properties from geophysical data. It can integrates petrophysical relationships and hydrological models with significant uncertainty in calibration parameters.

HBI decomposes the subsurface model into two components:

* **Groundwater Model**:

  * Predictable using petrophysical and hydrological modeling
  * Characterized by posterior probability density functions (PDFs)

* **Background Model**:

  * Not predictable from petrophysical or hydrological models
  * Represented by deterministic resistivity structures (1D/2D/3D)
  * Its uncertainty is not characterized

This decomposition allows computationally expensive Bayesian inference to be applied only to the groundwater component (which is lower-dimensional), while the background model is handled using conventional least-squares inversion.

### Expectation-Maximization Algorithm

To jointly estimate:

* The posterior PDFs of the **Groundwater Model**
* The maximum likelihood estimate (MLE) of the **Background Model**

HBI applies the **Expectation-Maximization (EM)** algorithm, where each iteration executes the following sequence:

* **E-step**: Bayesian inversion of the groundwater model using Metropolis-Hastings sampling.
* **M-step**: Least-squares inversion of the background model.

---

## How It Works

EMWTINV is executed via the `emwtinv_multi.py` script. Example applications are located in the `/Script` directory.

### Directory Structure

* `/Source`: Source code
* `/Script`: Example scripts for applying EMWTINV
* `/Utils`: Tools for plotting and analyzing results

### Configuration Files

Before running, place the following files in the same directory as `emwtinv_multi.py`:

## emwtinv_setup.txt

Defines input data, geophysical method, E-step and M-step parameters.

📁 Geophysical Data
* Data_list: Path to the observation file with geophysical data.
* Method: Geophysical method to be used. Options include:
     * DC2D – Electrical Resistivity Tomography (2D)
     * DC1D – Electrical Vertical Sounding
     * TEM1D – Transient Electromagnetics (1D)
     * MT1D – Magnetotellurics (1D, not tested)
     * GRAV2D – Gravity (2D, not tested)

🔧 M-STEP Parameters (Traditional Inversion of Background Model)
* externalmodel: Initial model flag (True/False)
* external_constant_model: Use a halfspace model as initial model (True/False)
* externalmodel_ref: Reference model flag (True/False)
* model_file: Path to the initial background model file (if externalmodel is True)
* model_mesh: Mesh file used for the initial background model
* modelref_file: Path to the reference model used during M-step regularization
* modelref_mesh: Mesh file used for the reference model
* invert_initial_bkmodel: Use traditional inversion to estimate the initial background model (True/False)
* reference_value: Value used for halfspace model when external_constant_model is True
* sub_lmd: Not used
* emmax_iter: Number of HBI iterations

🧮 SimPEG Regularization Parameters for Initial Background Model
* invparameters0.Collingrate: Number of iterations to reduce beta
* invparameters0.Beta: Trade-off factor
* invparameters0.Minbeta: Minimum beta
* invparameters0.Collingfactor: Factor to reduce beta
* invparameters0.alpha_x/y/z/s: Regularization weights for gradients and model smallness
* invparameters0.chifact: Misfit target

🧮 SimPEG Regularization Parameters for Background Model (M-Step)
* invparameters.Collingrate: Number of iterations to reduce beta
* invparameters.Beta: Trade-off factor
* invparameters.Minbeta: Minimum beta
* invparameters.Collingfactor: Factor to reduce beta
* invparameters.alpha_x/y/z/s: Regularization weights for gradients and model smallness
* invparameters.chifact: Misfit target

🔄 E-STEP Parameters (Bayesian Inversion of Groundwater Model)
* estep_niter: Number of groundwater model samples per MPI thread
* estep_beta: Not used
* estep_beta_factor: Not used
* estep_prop: Model generator type. Options:
   * REF – 2D unconfined aquifer constrained by water table (Zwt)
   * 2D – 2D unconfined aquifer (no Zwt)
   * 1D – 1D unconfined aquifer
* vmin: Not used
* vmax: Not used

# sigmawt_setup.txt
Defines the prior distributions and bounds for petrophysical parameters used to generate groundwater models. 

Example for 2D unconfined aquifer with ERT.

Parameter	Description
* Zmin	Minimum elevation of the water table at the center of the model
* Zmax	Maximum elevation of the water table at the center of the model
* dZ	Discretization step for water table elevation
* log_sigma_min	Minimum log bulk resistivity
* log_sigma_max	Maximum log bulk resistivity
* beta_min	Minimum CK-relationship correlation factor (β)
* beta_max	Maximum CK-relationship correlation factor (β)
* por_min	Minimum porosity
* por_max	Maximum porosity
* m1, m2	Range of Archie’s cementation factor (m)
* logA1, logA2	Range of log aquifer resistivity (log Ω·m) for Archie’s Law
* sigma0	Not used
* sigma0_var	Not used
* magic_lmd	Not used

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
