# Code for "Enhancing quantum variational state diagonalization using reinforcement learning techniques"

---

This is a repository for the code used to calculate the numerical results presented in the article "Enhancing quantum variational state diagonalization using reinforcement learning techniques".

---

## Software Installation
The code was used on Ubuntu OS 20.04.

For this project, we use Anaconda which can be downloaded from https://www.anaconda.com/products/individual.

To install activate the environment please do the following:

```
conda env create -f rl-vqsd.yml
conda activate rl-vqsd 
```
---
## How to generate quantum state to diagonalize?

The `utils.py` python script contains two important functions `random_state_gen(...)` and `ground_state_reduced_heisen_model(...)` corresponding to the generation of **(1)** An arbitrary quantum state sampled from the Haar measure. **(2)** The reduced ground state of the Heisenberg model. You just need to run
```
python utils.py --state state_typ --max_dim maxdim --seed seedno
```
**to generate (1) :** `state_typ` (`str`) is `arc-mixed`, `maxdim` (`int`) can be any upper limit to the size of the quantum state to be produced and `seedno` (`int`) specifies the seed for the quantum state.

**An Example** 
``` 
python utils.py --state arb-mixed --max_dim 4 --seed 1
```


**to generate (2) :** `state_typ` (`str`) is `reduce-heisen`, `maxdim` (`int`) is either 3 or 4 and `seedno` (`int`) is a redundant variable.

**An Example** 
``` 
python utils.py --state reduce-heisen --max_dim 3 --seed 342425
```

---
## How to run RL-VQSD?

To diagonalize a quantum state we can just run the `main.py` python script using the following line of code:

```
python main.py --seed seedagent --config config_file --experiment_name "global_COBYLA/"
```

In the above, the `seedagent` (`int`) corresponds to the different initialization to the Neural Network (NN) and the `config_file` (`str`) is the configuration corresponding to the state that need to be diagonalized, the hyperparameters of the NN and the agent configuration and it looks like: `h_s_2_rank_4_1`  where `2` corresponds to the number of qubit of the state, `4` is the rank of the state and `1` is the seed used to generate the state.

**An Example :** 
```
python main.py --seed 1 --config h_s_2_rank_4_1 --experiment_name "global_COBYLA/"
```

All the possible configurations can be found in the folder `configuration_files`.

**To run the reduced Heisenberg model :**

```
python main.py --seed 102 --config h_s_3_reduce_heisen --experiment_name "global_COBYLA/"
```

**Diagonalizing using random search :**

```
python main.py --seed 100 --config h_s_3_reduce_heisen --experiment_name "random_search/"
```

**Diagonalizing using Bootstrap DDQN :**

```
python main.py --seed 100 --config h_s_2_rank_4_1_bs --experiment_name "global_COBYLA/"
```

The `bs` in the suffix of each configuration file corresponds to the utilization of `Bootstrapping`.

## How to reproduce the results?
The results of the above will be saved in the `results` folder.

**The reproduce the 2-qubit eigenvalue convergence (Fig. 6a in article) :** You can run one of the jupyter notebooks titled `eigenvalue analysis.ipynb` and run each cell. Similarly,

**The reproduce the 3-qubit eigenvalue convergence (Fig. 9 in article) :** You can run one of the jupyter notebooks titled `eigenvalue analysis_reduce_heisen.ipynb` and run each cell.

**Constant structure RL-ansatz statistics (Fig. 8 in article) :** You just need to run the `constant_structure_VQSD.ipynb` to first load the RL-ansatz of your choice and then use this ansatz to diagonalize `N` arbitrary quantum states of same dimension. This is utilized to plot in the last couple of cells of the notebook.

**The reproduce the 2-qubit eigenvalue error (Fig. 6b in article) :** You first need to produce the diagonalization results using Layered Hardware Efficient Ansatz (LHEA) which can be done using and by running the `LHEA_VQSD.ipynb` file. Then utilizing the LHEA results in `LHEA_plot_analysis.ipynb` we produce Figure 6b.

**Comparison with random search (Fig. 11a, 11b and 12)** Both the plots for 2 and 3 qubits to compare the DDQN with random search can be generated just by running the `plot_analysis_random_search.ipynb` file. Finally,

**Comparison with Bootstrap DDQN (Fig. 13)** The comparison to Bootstrap DDQN can be produced/reproduced using `plot_analysis_BDDQN.ipynb`file. 