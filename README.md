# The DQCO Framework for UAWN Slicing - Simulation Code

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

This repository contains the simulation code for the paper:  
**"Dynamic Queue-Centric Optimization Framework for UAV-Assisted Wireless Network Slicing"**  
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

## 🚀 Quick Start
### Installation
```bash
git clone https://github.com/xtaiuiu/The-DQCO-Framework-for-UAWN-Slicing.git
cd The-DQCO-Framework-for-UAWN-Slicing
pip install -r requirements.txt
```

### Run Simulation
```bash
python scripts/run_simulation.py \
    --config data/config.yaml \
    --output results/simulation_1
```

### Reproduce Paper Figures
```python
from scripts.visualize import plot_throughput_comparison
plot_throughput_comparison("results/exp1/metrics.csv")
```

## 🔍 Key Features
- ✅ **Queue-aware optimization** for dynamic UAV networks
- ✅ **Multi-tenant slicing** with QoS guarantees
- ✅ **Gurobi/Pyomo** hybrid solver implementation
- ✅ **Monte Carlo simulation** with 500+ test cases

## 📊 Benchmark Results
| Metric | DQCO (Ours) | Baseline [1] |
|--------|-------------|-------------|
| Throughput | 12.7 Mbps | 9.2 Mbps |
| Latency | 28 ms | 45 ms |
| Slicing Accuracy | 92% | 83% |

## 📝 Citation
If you use this code in your research, please cite:
```bibtex
@article{dqco2024,
  title={Dynamic Queue-Centric Optimization for UAWN Slicing},
  author={Your Name, Coauthors},
  journal={IEEE Transactions on Wireless Communications},
  year={2024}
}
```

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first.

## 📧 Contact
For questions: [your.email@university.edu](mailto:your.email@university.edu)