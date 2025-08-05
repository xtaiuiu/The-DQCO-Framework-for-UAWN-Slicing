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