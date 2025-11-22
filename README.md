# 🧠 Neuro-Symbolic Causal Discovery (NSCD)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**NSCD** is a state-of-the-art hybrid framework for inferring causal relationships from time-series data. By fusing the non-linear pattern recognition of **Neural Networks** (LSTMs, GNNs) with the logical rigor of **Symbolic AI** (Constraint-based methods), it discovers interpretable and robust causal graphs.

---

## 🌟 Key Features

*   **Hybrid Architecture**: Combines PC Algorithm, Neural Granger Causality, and Graph Neural Networks.
*   **Symbolic Logic**: Enforces Sparsity and Acyclicity (DAG) constraints for valid structural discovery.
*   **Real-World Ready**: Proven on US Macroeconomic data and Meteorological datasets.
*   **Interactive Dashboard**: A Streamlit-based web app for instant visualization and storytelling.
*   **Stability Analysis**: Bootstrap-based confidence estimation for every discovered edge.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/nscd.git
cd nscd
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
Explore our pre-loaded Weather Analysis or upload your own data:
```bash
streamlit run app.py
```
*The app will automatically process the included `yoyo.csv` weather dataset on first run.*

### 3. Run Benchmarks
Reproduce our research results on US Macroeconomic data:
```bash
python notebooks/benchmark.py
```

---

## 🔬 How It Works

The NSCD pipeline operates in 4 stages:

1.  **Symbolic Skeleton (PC Algorithm)**:
    *   Uses conditional independence tests to prune the search space and find an initial undirected skeleton.
2.  **Neural Orientation (LSTM Granger)**:
    *   Trains LSTM networks to detect non-linear, time-lagged dependencies between variable pairs.
3.  **Global Refinement (Causal GNN)**:
    *   A Graph Neural Network refines edge probabilities by aggregating global structural information.
4.  **Logic Constraints**:
    *   Applies domain-agnostic rules (e.g., "No Cycles") to ensure the final output is a valid Causal DAG.

---

## 📊 Results

### 🌤️ Weather Systems (Yoyo Dataset)
*   **Solar Radiation $\to$ Temperature**: Recovered the primary driver of thermal energy.
*   **Rain $\to$ Humidity**: Identified precipitation as a direct cause of humidity increases.

### 📈 US Macroeconomics
*   **Interest Rates $\to$ Investment $\to$ GDP**: Validated the monetary transmission mechanism.
*   **Unemployment $\to$ Inflation**: Captured the Phillips Curve trade-off.

---

## 📂 Project Structure

```
.
├── app.py                  # Streamlit Dashboard
├── notebooks/              # Benchmarking & Processing Scripts
│   ├── benchmark.py        # US Macro Analysis
│   └── process_yoyo.py     # Weather Data Processing
├── src/                    # Core Source Code
│   ├── pipeline.py         # Unified NSCD Pipeline
│   ├── models/             # PC, LSTM, GNN Implementations
│   ├── constraints/        # Symbolic Logic Rules
│   ├── evaluation/         # Stability & Visualization
│   └── data/               # Preprocessing Utils
├── results/                # Generated Graphs & Pickle Files
└── paper/                  # Research Paper (Markdown)
```

## 📜 Citation

If you use this code in your research, please cite:

> **Neuro-Symbolic Causal Discovery for Time-Series Dynamics**  
> *AI Research Initiative, 2025.*

---

*Built with ❤️ by Antigravity*
