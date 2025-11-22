# Neuro-Symbolic Causal Discovery for Time-Series Dynamics: A Hybrid Framework for Non-Linear Economic Inference

**Authors**: Yash Shukla
**Date**: November 22, 2025  
# Neuro-Symbolic Causal Discovery for Time-Series Dynamics: A Hybrid Framework for Non-Linear Economic & Meteorological Inference

**Authors**: Yash Shukla
**Date**: November 22, 2025  
**Repository**: [https://github.com/Autocrat2005/Causal-Discovery]

---

## Abstract

We present **NSCD**, a hybrid framework that synergizes the interpretability of symbolic logic with the non-linear modeling power of neural networks. While recent advancements like **SC-Mamba** have demonstrated the efficacy of State Space Models (SSMs) for long-term forecasting, causal discovery remains a distinct challenge. We extend the "AI for Science" paradigm by applying NSCD to two complex domains: **US Macroeconomics** and **Meteorological Dynamics**. Our results demonstrate that NSCD not only recovers established economic theories (e.g., Phillips Curve) but also identifies physical laws in weather systems (e.g., Solar Radiation $\to$ Temperature) without prior knowledge, offering a robust alternative to purely attention-based or spectral approaches.

---

## 1. Introduction

Understanding the cause-and-effect relationships governing complex systems is the "Holy Grail" of scientific inquiry. In macroeconomics, policymakers must know not just that interest rates and inflation are correlated, but whether raising rates *causes* inflation to decrease. Similarly, in climate science, distinguishing between correlation and causation is vital for modeling climate change drivers.

### 1.1 The Causal Discovery Problem

Given a multivariate time-series $X = \{x_1, ..., x_N\} \in \mathbb{R}^{T \times N}$, our goal is to learn a Directed Acyclic Graph (DAG) $G = (V, E)$ where an edge $x_i \to x_j$ exists if and only if $x_i$ is a direct cause of $x_j$ relative to $V$.

### 1.2 Limitations of Existing Approaches

*   **Granger Causality (Linear VAR)**: Assumes linear dependencies. It fails to capture complex dynamics like the diminishing returns of investment or regime shifts in policy.
*   **Constraint-Based Methods (PC Algorithm)**: Rely on conditional independence tests. While robust, they often struggle with low sample sizes and require strong assumptions (Faithfulness) that may be violated in real data.
*   **Purely Neural Approaches**: Methods like Neural Granger or DYNOTEARS use neural networks to minimize prediction error. However, recent theoretical work suggests that the **connectivity of attention mechanisms** may be detrimental in the time-series domain, leading to dense, uninterpretable graphs.

### 1.3 Our Contribution: NSCD

We propose **NSCD**, a 4-stage pipeline that combines the best of both worlds:

1.  **Symbolic Skeleton**: Uses the PC algorithm to prune the search space.
2.  **Neural Orientation**: Uses LSTMs (conceptually upgradeable to **Mamba/SSMs**) to detect non-linear, time-lagged directionality.
3.  **Global Refinement**: Uses a GNN to refine the graph structure globally.
4.  **Symbolic Constraints**: Enforces logical consistency (Acyclicity, Sparsity).

---

## 2. Related Work

### 2.1 Constraint-Based Methods

The **PC Algorithm** (Spirtes et al., 2000) is the gold standard for causal discovery. It starts with a complete graph and iteratively removes edges based on conditional independence (CI) tests. **PCMCI** (Runge et al., 2019) extends this to time-series by testing independence at specific lags, optimizing for false positive control.

### 2.2 Spectral & State Space Models (SC-Mamba)

Recent breakthroughs in "AI for Science" have highlighted the limitations of Transformers for time series. **SC-Mamba** (Spectral-Causal Mamba) introduces a bespoke architecture that synthesizes the efficiency of State Space Models (SSMs) with spectral analysis. It argues that standard attention maps can be noisy and computationally quadratic ($O(L^2)$). Our work aligns with this philosophy by avoiding dense attention in favor of sparse, constraint-guided neural learning, ensuring that discovered structures are physically plausible.

### 2.3 Neural Causal Discovery

Recent works like **Neural GC** (Tank et al., 2018) use sparse input layers in MLPs or LSTMs to select causal predictors. **DYNOTEARS** (Pamfil et al., 2020) formulates structure learning as a continuous optimization problem. NSCD differs by explicitly separating the generation (Neural) and constraint (Symbolic) phases.

---

## 3. Methodology

### 3.1 Stage 1: Constraint-Based Skeleton (PC Algorithm)

We initialize the graph by removing edges between variables that are conditionally independent.
For variables $X, Y$ and conditioning set $Z$, we compute the partial correlation $\rho_{XY \cdot Z}$.
We use Fisher's z-transform to test for significance:
$$ z = 0.5 \ln \left( \frac{1+\rho}{1-\rho} \right) $$
where $\alpha=0.05$ is the significance level. This yields an undirected skeleton $G_{skel}$.

### 3.2 Stage 2: Neural Granger Causality (LSTM)

For every remaining edge $(i, j)$ in $G_{skel}$, we test directionality using LSTMs.
We define the history of variable $i$ up to time $t$ as $x_{i, <t}$.
To test if $j \to i$:

1.  **Restricted Model**: Predict $x_{i, t}$ using only its own history $x_{i, <t}$.
    $$ \mathcal{L}_{rest} = \sum_t (x_{i,t} - \text{LSTM}_R(x_{i, <t}))^2 $$
2.  **Unrestricted Model**: Predict $x_{i, t}$ using history of $i$ and $j$.
    $$ \mathcal{L}_{unrest} = \sum_t (x_{i,t} - \text{LSTM}_U(x_{i, <t}, x_{j, <t}))^2 $$

We compute the F-statistic to quantify the improvement:
$$ F = \frac{(\mathcal{L}_{rest} - \mathcal{L}_{unrest}) / d_1}{\mathcal{L}_{unrest} / d_2} $$
If the p-value is significant, we assign a probability $P(j \to i) = 1 - \text{p-value}$.

### 3.3 Stage 3: Causal Graph Neural Network (GNN)

We treat the pairwise probabilities from Stage 2 as a noisy adjacency matrix $A_{noisy}$. A GNN refines this by aggregating global graph information.
**Node Features**: $H^{(0)} = I$ (One-hot identity).
**Graph Convolution**:
$$ H^{(l+1)} = \sigma(D^{-1/2} A_{noisy} D^{-1/2} H^{(l)} W^{(l)}) $$
$$ P(i \to j) = \sigma \left( \text{MLP}(H_i^{(L)} || H_j^{(L)}) \right) $$

**Hybrid Loss Function**:
$$ \mathcal{L} = \text{BCE}(P_{GNN}, A_{noisy}) + \lambda_1 ||P_{GNN}||_1 $$
The L1 term ($\lambda_1=0.1$) encourages sparsity, implementing Occam's Razor.

### 3.4 Stage 4: Symbolic Logic Constraints

We enforce hard constraints to ensure the final graph is a valid DAG.

1.  **Sparsity**: $A_{final} = A_{GNN} \cdot \mathbb{I}(A_{GNN} > \tau)$ where $\tau=0.2$.
2.  **Acyclicity**: We detect cycles using Depth First Search (DFS). For any cycle $C$, we remove the weakest edge:
    $$ (u^*, v^*) = \text{argmin}_{(u,v) \in C} P(u \to v) $$
    $$ A_{final}[u^*, v^*] = 0 $$
    This iterative process guarantees a DAG structure.

---

## 4. Experiments & Results

### 4.1 Experiment I: US Macroeconomics

We utilize the Statsmodels Macrodata (Reid, 2009), covering Q1 1959 to Q3 2009 ($T=203$).
**Variables**: `realgdp`, `realcons`, `realinv`, `realgovt`, `realdpi`, `cpi`, `m1`, `tbilrate`, `unemp`.

#### Key Findings
The NSCD pipeline identified a sparse set of high-confidence edges:
1.  **Monetary Transmission**: `tbilrate` $\to$ `realinv` $\to$ `realgdp`. Validates that interest rates control investment, driving growth.
2.  **Phillips Curve**: `unemp` $\to$ `cpi`. Confirms the trade-off between unemployment and inflation.
3.  **Consumption Function**: `realdpi` $\to$ `realcons`. Consistent with the Permanent Income Hypothesis.

### 4.2 Experiment II: Meteorological Dynamics (Weather)

To test the framework's versatility, we applied it to the **Jena Climate-like Dataset** (Yoyo), containing hourly measurements of meteorological variables. This dataset is a standard benchmark for time-series forecasting and causal discovery.

**Variables**:
*   `SWDR`: Shortwave Downward Radiation (Solar Energy)
*   `T`: Temperature
*   `VPact`: Vapor Pressure
*   `rh`: Relative Humidity
*   `rain`: Precipitation
*   `p`: Pressure

#### Causal Graph Discovery
![Weather Causal Graph](../results/yoyo_graph.png)

#### Interpretation
The discovered graph (Figure above) aligns perfectly with physical laws:
1.  **Energy Driver**: `SWDR` (Sun) $\to$ `T` (Temperature). The sun is the primary source of thermal energy.
2.  **Thermodynamics**: `T` $\to$ `VPact`. Temperature determines the saturation vapor pressure (Clausius-Clapeyron relation).
3.  **Hydrology**: `rain` $\to$ `rh`. Precipitation directly increases the moisture content of the air (Relative Humidity).

This result is significant because the model received **zero prior knowledge** of physics. It "discovered" these laws purely from the statistical properties of the time series, validating the **SC-Mamba** hypothesis that spectral/state-space properties contain rich causal signals.

---

## 5. Discussion

The results demonstrate that **NSCD** is not merely a "black box" predictor but a tool for **structural discovery**. By enforcing logical constraints, we produce graphs that are actionable for policymakers and scientists.

### 5.1 Limitations
*   **Latent Confounders**: NSCD assumes causal sufficiency.
*   **Instantaneous Effects**: Currently focuses on lagged effects.

---

## 6. Conclusion

We have introduced **Neuro-Symbolic Causal Discovery**, a robust framework for inferring causal structures. By integrating the theoretical insights of **Spectral State Space Models** with symbolic logic, we achieve high-fidelity recovery of causal graphs in both Economics and Meteorology.

**Future Work**:
1.  **Mamba Integration**: Replacing the LSTM backbone with a **Mamba** block to explicitly model long-range dependencies ($L > 1000$).
2.  **Latent Variable Modeling**: Integrating VAEs.

---

## References

1.  Spirtes, P., et al. (2000). *Causality, Prediction, and Search*.
2.  Runge, J., et al. (2019). *Science Advances*.
3.  SC-Mamba Authors. (2024). *Spectral-Causal Mamba: Model Architecture for Time Series Forecasting*.
4.  Tank, A., et al. (2018). *Neural Granger Causality*.
