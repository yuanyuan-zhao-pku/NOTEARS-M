# NOTEARS-M: Causal Bayesian Network Structure Learning of Mixed Type Data

This repository contains the implementation of the NOTEARS-M algorithm, as well as the simulation data, code, and results associated with the manuscript submitted to **BMC Medical Research Methodology**. The repository is organized into nine subfolders, each containing specific files and results related to the algorithm and the simulation scenarios described in the manuscript.

---

## Repository Structure

### 1. **notearsm**
This folder contains the core implementation of the NOTEARS-M algorithm:
- **`NOTEARS_M.py`**: The Python script implementing the NOTEARS-M algorithm for learning Directed Acyclic Graph (DAG) structures.

### 2. **simu-numnodes**
This folder corresponds to the simulation scenario investigating the effect of the **number of nodes** in the graph. It includes:
- **`B_true_*.csv`**: The true adjacency matrices for the simulations.
- **`W_true_*.csv`**: The true weighted adjacency matrices for the simulations.
- **`simulation-numnodes.py`**: The script for generating simulation data for this simulation.
- **`notearsm-simulation-numnodes.py`**: The script for applying the NOTEARS-M algorithm to learn DAG structures in this simulation.
- **`each-accuracy-calculator-numnodes.py`**: The script for calculating accuracy metrics for each replication.
- **`Comparison methods.R`**: The R script implementing three comparison methods (MMHC, mDAG, DAGBagM) for this simulation.
- **`sem_type.txt`**: A file specifying the type of each node in the simulation.
- Subfolders for each method:
  - **`MMHC`**, **`mDAG`**, **`DAGBagM`**, **`NOTEARS-M`**: Each subfolder contains:
    - **`W_est_*.csv`**: The estimated weighted adjacency matrices for each replication.
    - **`each_acc_*.csv`**: Accuracy metrics for each replication.
    - **`acc_*.csv`**: Average accuracy metrics across 10 replications.
    - **`X_*.csv`**: The generated data files for each replication.

### 3. **simu-catratio**
This folder corresponds to the simulation scenario investigating the effect of the **proportion of categorical nodes**. It includes the same types of files and subfolders as described in **simu-numnodes**, but specific to this simulation.

### 4. **simu-samplesize**
This folder corresponds to the simulation scenario investigating the effect of **sample size**. It includes the same types of files and subfolders as described in **simu-numnodes**, but specific to this simulation.

### 5. **simu-catnum**
This folder corresponds to the simulation scenario investigating the effect of the **levels of categorical nodes**. It includes the same types of files and subfolders as described in **simu-numnodes**, but specific to this simulation.

### 6. **simu-s0**
This folder corresponds to the simulation scenario investigating the effect of **edge sparsity**. It includes the same types of files and subfolders as described in **simu-numnodes**, but specific to this simulation.

### 7. **simu-walpha**
This folder corresponds to the simulation scenario investigating the effect of **weight scale**. It includes the same types of files and subfolders as described in **simu-numnodes**, but specific to this scenario.

### 8. **simu-graphmodel**
This folder corresponds to the simulation scenario investigating the effect of **graph type**. It includes the same types of files and subfolders as described in **simu-numnodes**, but specific to this simulation.

### 9. **simu-distribution**
This folder corresponds to the simulation scenario investigating the effect of **noise distribution**. It includes the same types of files and subfolders as described in **simu-numnodes**, but specific to this simulation.

---

## File Descriptions

### Core Algorithm
- **`NOTEARS_M.py`**: The Python implementation of the NOTEARS-M algorithm, which extends the original NOTEARS algorithm (Xun Zheng et al.) to handle mixed data types (continuous and categorical).

### Simulation Data and Code
Each simulation folder contains:
1. **True Graph Files**:
   - **`B_true_*.csv`**: True adjacency matrices for the simulated graphs.
   - **`W_true_*.csv`**: True weighted adjacency matrices for the simulated graphs.
2. **Simulation Scripts**:
   - **`simulation-*.py`**: Scripts for generating simulation data for each scenario.
   - **`notearsm-simulation-*.py`**: Scripts for applying the NOTEARS-M algorithm to learn DAG structures in each scenario.
3. **Accuracy Calculation**:
   - **`each-accuracy-calculator-*.py`**: Scripts for calculating accuracy metrics (e.g., Structural Hamming Distance, True Positive Rate, etc.) for each replication.
4. **Comparison Methods**:
   - **`Comparison methods.R`**: R scripts implementing three comparison methods (MMHC, mDAG, DAGBagM) for each scenario.
5. **Node Type File**:
   - **`sem_type.txt`**: Specifies the type (categorical or continuous) of each node in the simulation.
6. **Generated Data and Results**:
   - **`X_*.csv`**: Simulated data files for each replication.
   - **`W_est_*.csv`**: Estimated weighted adjacency matrices for each replication.
   - **`each_acc_*.csv`**: Accuracy metrics for each replication.
   - **`acc_*.csv`**: Average accuracy metrics across 10 replications.

---

## How to Use the Repository

1. **Run the NOTEARS-M Algorithm**:
   - Use the **`NOTEARS_M.py`** script to apply the NOTEARS-M algorithm to your own data or the provided simulation data.

2. **Reproduce Simulation Results**:
   - Navigate to the relevant simulation folder (e.g., **simu-numnodes**, **simu-catratio**, etc.).
   - Run the **`simulation-*.py`** script to generate simulation data.
   - Run the **`notearsm-simulation-*.py`** script to apply the NOTEARS-M algorithm to the generated data.
   - Use the **`each-accuracy-calculator-*.py`** script to calculate accuracy metrics for each replication.

3. **Compare with Other Methods**:
   - Use the **`Comparison methods.R`** script to apply MMHC, mDAG, and DAGBagM to the same simulation data.
   - Compare the results stored in the corresponding subfolders (**MMHC**, **mDAG**, **DAGBagM**, **NOTEARS-M**).

---

## Requirements

### Python
- Python 3.8 or higher
- Required libraries: `numpy`, `scipy`, `pandas`, `igraph`, `sklearn`

### R
- R 4.0 or higher
- Required packages: `bnlearn`, `mDAG`, `DAGBagM`

---

## Citation

If you use the NOTEARS-M algorithm or the simulation data in your research, please cite our manuscript:

[Full citation details will be added upon publication.]

---

## Contact

For any questions or issues regarding the code or data, please contact:

- **Yuanyuan Zhao**: [zhaoyuanyuan1@pku.edu.cn]
- **Jinzhu Jia**: [jzjia@math.pku.edu.cn]

---

