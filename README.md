# The DQCO Framework for UAWN Slicing - Simulation Code

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

This repository contains the simulation code for the paper:  
**"Fast Slicing for UAV-Aided Wireless Networks: A
Quasiconvex Decomposition Framework"**  
*(Submitted to IEEE INFOCOM, 2026)*

## 📜 Repository Structure
```bash
.
├── core/                  # Core algorithm implementations
│   ├── dqco_engine.py     # Main optimization engine
│   └── slicing_model.py   # Network slicing models
├── data/                  # Input datasets/configs
│   ├── uav_positions.json
│   └── traffic_profiles.h5
├── scripts/               # Execution scripts
│   ├── run_simulation.py  # Main simulation script
│   └── visualize.py       # Result visualization
├── results/               # Output figures/data (auto-generated)
├── requirements.txt       # Python dependencies
```

<!-- TREEVIEW START -->
.
├── .idea
│   ├── inspectionProfiles
│   │   └── profiles_settings.xml
│   ├── .gitignore
│   ├── misc.xml
│   ├── modules.xml
│   ├── vcs.xml
│   └── 仿真.iml
├── algorithms
│   ├── experiment_comparison
│   │   ├── __init__.py
│   │   ├── compare_num_UEs.py
│   │   ├── compare_theta.py
│   │   ├── compare_total_bandwidth.py
│   │   ├── compare_total_power.py
│   │   ├── convergence_function_values.xlsx
│   │   ├── objective_value_bandwidths_20times.xlsx
│   │   ├── objective_value_bandwidths_2times.xlsx
│   │   ├── objective_value_bandwidths_jnl.xlsx
│   │   ├── objective_value_num_UEs_20times.xlsx
│   │   ├── objective_value_num_UEs_2times.xlsx
│   │   ├── objective_value_num_UEs_jnl.xlsx
│   │   ├── objective_value_powers_jnl.xlsx
│   │   ├── objective_value_radius_jnl.xlsx
│   │   └── problem_generator.py
│   ├── experiment_convergence
│   │   ├── pso_iterations
│   │   │   ├── goc.pdf
│   │   │   ├── goc.png
│   │   │   ├── pso_convergence.txt
│   │   │   └── pso_convergence.xlsx
│   │   ├── shio_iterations
│   │   │   ├── goc.pdf
│   │   │   ├── goc.png
│   │   │   ├── shio_convergence.txt
│   │   │   └── shio_convergence.xlsx
│   │   ├── __init__.py
│   │   ├── convergence_function_values.xlsx
│   │   ├── convergence_time.py
│   │   ├── convergence_time.xlsx
│   │   ├── convergence_time_jnl.py
│   │   ├── convergence_time_summation.xlsx
│   │   ├── cvx_runtime.xlsx
│   │   ├── kkt_runtime.xlsx
│   │   ├── num_iterations.xlsx
│   │   ├── num_iterations_different_UE.xlsx
│   │   ├── num_of_iterations.py
│   │   ├── runtime_bcd_compare_jnl.xlsx
│   │   ├── runtime_benchmark_compare_jnl.xlsx
│   │   ├── runtime_obj_compare_jnl.xlsx
│   │   └── runtime_time_compare_jnl.xlsx
│   ├── experiment_rounding
│   │   ├── __init__.py
│   │   ├── convergence_function_values.xlsx
│   │   ├── obj_compare_unused.py
│   │   ├── rounding_compare_bandwidth_mean.py
│   │   ├── rounding_compare_bandwidth_variance.py
│   │   ├── rounding_compare_nUEs.py
│   │   ├── rounding_fairness.xlsx
│   │   ├── rounding_fairness_ton.xlsx
│   │   ├── rounding_objective.xlsx
│   │   ├── rounding_objective_ton.xlsx
│   │   ├── scenario_2_slices.pickle
│   │   └── total_throughput_compare.py
│   ├── main_algorithms
│   │   ├── block_descent_nonsmooth
│   │   │   ├── __init__.py
│   │   │   ├── block_descent_max_min_rate.py
│   │   │   ├── convergence_function_values.xlsx
│   │   │   ├── cvx_subproblem_xp.py
│   │   │   ├── scenario_2_slices.pickle
│   │   │   ├── subproblem_p.py
│   │   │   └── subproblem_x.py
│   │   ├── rounding
│   │   │   ├── __init__.py
│   │   │   ├── problem_generator.py
│   │   │   └── rounding_algorithms.py
│   │   ├── BCD_for_SUM_obj.py
│   │   ├── CGP.py
│   │   ├── __init__.py
│   │   ├── convergence_function_values.xlsx
│   │   ├── convergence_function_values_30 UE.xlsx
│   │   ├── convergence_function_values_60 UE.xlsx
│   │   ├── func_val_gradient.py
│   │   ├── geneticalgorithm.py
│   │   ├── mip_bcd.py
│   │   ├── problem.pickle
│   │   ├── projection.py
│   │   ├── quasi_subgradient_no_QoS.py
│   │   ├── quasi_subgradient_no_QoS_dynamic.py
│   │   ├── relax_quasi_subgradient.py
│   │   ├── scenario_2_UEs.pickle
│   │   ├── scenario_2_slices.pickle
│   │   ├── subgradient_for_2_dims.py
│   │   └── subgradient_test.py
│   └── __init__.py
├── matlab_code
│   └── SimplexProj.m
├── network_classes
│   ├── Variables.py
│   ├── __init__.py
│   ├── ground_user.py
│   ├── network_slice.py
│   ├── physical_network.py
│   ├── scenario.py
│   └── uav.py
├── scenarios
│   ├── UE_creators.py
│   ├── UE_distributions.pdf
│   ├── UE_set_2024-04-08_17-47-03.pkl
│   ├── UE_set_2025-08-05_11-12-29.pkl
│   ├── UE_set_2025-08-05_11-14-52.pkl
│   ├── __init__.py
│   ├── physical_network_creators.py
│   ├── scenario_2_UEs.pickle
│   ├── scenario_creators.py
│   └── slice_creators.py
├── test
│   ├── bcd_max_min_test
│   │   ├── __init__.py
│   │   ├── convergence_function_values.xlsx
│   │   ├── max_min_bcd_test.py
│   │   ├── problem_p_test.py
│   │   └── problem_x_test.py
│   ├── rounding_test
│   │   ├── __init__.py
│   │   └── problem_generator_test.py
│   └── __init__.py
├── utils
│   ├── __init__.py
│   ├── aoe_method.py
│   ├── cvx_test.py
│   ├── fairness.py
│   ├── max_min_knapsack.py
│   ├── problem.pickle
│   ├── projction.py
│   ├── scenario_2_slices.pickle
│   ├── test.xlsx
│   ├── test_func_grad.py
│   └── timer_test.py
├── README.md
├── requirements.txt
└── setup.py

19 directories, 125 files
<!-- TREEVIEW END -->

## 🚀 Quick Start
### Installation
```bash
pip install -r requirements.txt
```

### Run Simulation

#### 1. Create a simulated UAWN

#### 2. Solve the UAV Slicing Problem (USP) by using the DQCO framework


#### 3. Plot the Convergence Curve

```bash
python scripts/run_simulation.py \
    --config data/config.yaml \
    --output results/simulation_1
```

### Reproduce Paper Figures

##### Simulation requires time. To see the simulated figures, run the following codes.

```python
from scripts.visualize import plot_throughput_comparison
plot_throughput_comparison("results/exp1/metrics.csv")
```


## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.
