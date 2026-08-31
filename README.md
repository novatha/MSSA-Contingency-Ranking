# A Risk-Based Modified Salp Swarm Algorithm for N-k Contingency Ranking in Power Systems

[![Language: Julia](https://img.shields.io/badge/Language-Julia_1.9+-9558B2.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![IEEE Compliance](https://img.shields.io/badge/IEEE-Reproducible_Research-00629B.svg)](https://journals.ieeeauthorcenter.ieee.org/)

Official computational source code for the research paper:  
**"A Risk-Based Modified Salp Swarm Algorithm for N-k Contingency Ranking in Power Systems"**  
*Authors: Novalio Daratha, Fitra Akbar, Adhadi Kurniawan, and Hendy Santosa*  
*Department of Electrical Engineering, Faculty of Engineering, Universitas Bengkulu, Indonesia*

---

## ⚡ Overview

This repository provides high-performance Julia implementations of the **Direct On-The-Fly Discrete Modified Salp Swarm Algorithm (MSSA)** for power system $N-k$ security assessment and contingency risk ranking.

### Key Capabilities:
- **Zero Pre-computation Memory Bottleneck:** Searches directly in discrete branch-index tuple space with dynamic on-the-fly nonlinear AC power flow evaluation.
- **Probabilistic Risk Index:** Systematically combines voltage severity ($IP_V$), line overload ($IP_F$), and joint outage probability ($P_j$).
- **Multi-System Support:** Standard IEEE systems (14, 30, 57, 118, 300, 1,354-bus PEGASE) and the real-world 520-bus Sumatra transmission grid.
- **Competitor Baseline Algorithms:** Includes enhanced benchmark implementations for Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO).

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Install [Julia](https://julialang.org/downloads/) (version 1.9 or higher).

### 2. Environment Setup
Clone or navigate to this folder and instantiate the Julia dependencies:
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
*Required Packages:* `PowerModels.jl`, `Ipopt.jl`, `JuMP.jl`, `Combinatorics.jl`.

---

## 🧪 Running Benchmarks & Experiments

### Quick Single Test (e.g., IEEE 57-Bus, $N-2$ Outages):
```bash
julia --threads=auto run_benchmarks.jl case57 2
```

### Real-World Grid Test (Sumatra 520-Bus Grid, 292,230 $N-2$ Contingencies):
```bash
julia --threads=auto run_benchmarks.jl casesumatera 2
```
*Expected execution time: $\approx$ 9 seconds with 100% top-20 capture ratio.*

### Higher-Order $N-3$ Contingency Test:
```bash
julia --threads=auto run_benchmarks.jl case30 3
julia --threads=auto run_benchmarks.jl case57 3
```

### Full Automated Benchmark Suite:
```bash
julia --threads=auto run_benchmarks.jl --all
```

---

## 📁 Repository Structure

```
code_reproducibility/
├── MSSA_direct_nk.jl          # Direct On-The-Fly MSSA implementation (Core Algorithm)
├── MSSA_nk.jl                 # Two-stage vectorised MSSA optimizer
├── GA_nk.jl                   # Enhanced Genetic Algorithm baseline
├── PSO_nk.jl                  # Enhanced Particle Swarm Optimization baseline
├── ACO_nk.jl                  # Enhanced Ant Colony Optimization baseline
├── analisis_nk_optimized_final.jl # AC Power Flow solver with divergence/islanding penalty
├── contingency_utils.jl       # Probability calculation & repair operators
├── run_benchmarks.jl          # Unified CLI benchmark runner
├── Project.toml               # Package dependencies and version constraints
├── README.md                  # Reproduction instructions and documentation
└── test_cases/                # Power grid benchmark models
    ├── case14.m
    ├── case30.m
    ├── case57.m
    ├── case118.m
    ├── case300.m
    ├── case1354pegase.m
    └── casesumatera.m
```

---

## 🔬 Mathematical Formulation

The objective is to identify the most critical $N-k$ contingency set $\mathcal{S}$ within the operational monitoring budget $C$:

$$\max_{\mathcal{S} \subset \Omega_k, \, |\mathcal{S}| \le C} \sum_{j \in \mathcal{S}} \text{RI}_j$$

where the **Risk Index (RI)** is defined as:
$$\text{RI}_j = P_j \times \text{CPI}_j$$
$$\text{CPI}_j = w_V \cdot \text{IP\_V}_j + w_F \cdot \text{IP\_F}_j$$

* **Voltage Severity ($IP_V$):**
  $$\text{IP\_V}_j = \sum_{i=1}^{N_b} \left( \frac{V_{i,j} - V_i^{\text{lim}}}{V_i^{\text{max}} - V_i^{\text{lim}}} \right)^{2m}$$
* **Divergence / Voltage Collapse Penalty:** Divergent power flows or islanded subsystems without generation are assigned $\text{RI} = 10^{10} \times P_j$ to prioritize catastrophic events.

---

## 📜 Funding & Acknowledgments

This research was funded by the **Faculty of Engineering, Universitas Bengkulu**, under the Flagship Research Grant Scheme for Lecturers (*Penelitian Unggulan Bagi Dosen FT UNIB*), Contract No. **`7064/UN30.13/PG/2025`**.

---

## 📖 Citation

If you use this codebase in your research, please cite our paper:

```bibtex
@article{daratha2026mssa,
  author    = {Daratha, Novalio and Akbar, Fitra and Kurniawan, Adhadi and Santosa, Hendy},
  title     = {A Risk-Based Modified Salp Swarm Algorithm for N-k Contingency Ranking in Power Systems},
  journal   = {IEEE Transactions on Power Systems},
  year      = {2026},
  note      = {Under Review}
}
```

---

## ⚖️ License
This project is open-source and available under the **MIT License**.
