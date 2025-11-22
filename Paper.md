# Neuro-Symbolic Causal Discovery for Time-Series Dynamics: A Hybrid Framework for Non-Linear Economic Inference

**Authors**: Yash Shukla
**Date**: November 22, 2025  
**Repository**: [https://github.com/Autocrat2005/Causal-Discovery]

---

## Abstract

Causal discovery from observational time-series data remains a formidable challenge in econometrics and systems science. Traditional constraint-based methods often rely on restrictive linearity assumptions, while modern deep learning approaches, though capable of modeling non-linearities, frequently lack interpretability and structural validity. We present the **Neuro-Symbolic Causal Discovery (NSCD)** framework, a novel hybrid system designed to infer complex, non-linear causal relationships from multivariate time-series data. NSCD synergizes the pattern recognition capabilities of Long Short-Term Memory (LSTM) networks and Graph Neural Networks (GNNs) with the logical rigor of symbolic constraint enforcement. We evaluate NSCD on a comprehensive dataset of 9 US Macroeconomic indicators (1959-2009), uncovering a complex, interpretable web of causal interactions—including Monetary Policy transmission mechanisms and the Phillips Curve—that aligns with established economic theory while revealing non-linear nuances missed by linear baselines.

---

## 1. Introduction

Understanding the cause-and-effect relationships governing complex systems is the "Holy Grail" of scientific inquiry. In macroeconomics, policymakers must know not just that interest rates and inflation are correlated, but whether raising rates _causes_ inflation to decrease.

### 1.1 The Causal Discovery Problem

Given a multivariate time-series $X = \{x_1, ..., x_N\} \in \mathbb{R}^{T \times N}$, our goal is to learn a Directed Acyclic Graph (DAG) $G = (V, E)$ where an edge $x_i \to x_j$ exists if and only if $x_i$ is a direct cause of $x_j$ relative to $V$.

### 1.2 Limitations of Existing Approaches

- **Granger Causality (Linear VAR)**: Assumes linear dependencies. It fails to capture complex dynamics like the diminishing returns of investment or regime shifts in policy.
- **Constraint-Based Methods (PC Algorithm)**: Rely on conditional independence tests. While robust, they often struggle with low sample sizes and require strong assumptions (Faithfulness) that may be violated in real data.
- **Neural Approaches**: Methods like Neural Granger or DYNOTEARS use neural networks to minimize prediction error. However, they often produce dense, cyclic graphs that violate the DAG assumption, making them physically impossible interpretations of causality.

### 1.3 Our Contribution: NSCD

We propose **NSCD**, a 4-stage pipeline that combines the best of both worlds:

1.  **Symbolic Skeleton**: Uses the PC algorithm to prune the search space.
2.  **Neural Orientation**: Uses LSTMs to detect non-linear, time-lagged directionality.
3.  **Global Refinement**: Uses a GNN to refine the graph structure globally.
4.  **Symbolic Enforcement**: Applies logic rules to guarantee a valid, sparse DAG.

---

## 2. Related Work

### 2.1 Constraint-Based Methods

The **PC Algorithm** (Spirtes et al., 2000) is the gold standard for causal discovery. It starts with a complete graph and iteratively removes edges based on conditional independence (CI) tests. **PCMCI** (Runge et al., 2019) extends this to time-series by testing independence at specific lags, optimizing for false positive control.

### 2.2 Functional Causal Models

Methods like **LiNGAM** assume non-Gaussian noise to identify directionality in linear models. **Post-Non-Linear (PNL)** models extend this to specific non-linear forms.

### 2.3 Neural Causal Discovery

Recent works like **Neural GC** (Tank et al., 2018) use sparse input layers in MLPs or LSTMs to select causal predictors. **DYNOTEARS** (Pamfil et al., 2020) formulates structure learning as a continuous optimization problem with an acyclicity constraint trace term. NSCD differs by explicitly separating the generation (Neural) and constraint (Symbolic) phases, allowing for more flexible architecture choices.

---

## 3. Methodology

### 3.1 Stage 1: Constraint-Based Skeleton (PC Algorithm)

We initialize the graph by removing edges between variables that are conditionally independent.
For variables $X, Y$ and conditioning set $Z$, we compute the partial correlation $\rho_{XY \cdot Z}$.
We use Fisher's z-transform to test for significance:
$$ z = \frac{1}{2} \ln \left( \frac{1 + \rho}{1 - \rho} \right) $$
$$ \sqrt{T - |Z| - 3} \cdot |z| > \Phi^{-1}(1 - \alpha/2) \implies X \not\perp Y | Z $$
where $\alpha=0.05$ is the significance level. This yields an undirected skeleton $G_{skel}$.

### 3.2 Stage 2: Neural Granger Causality (LSTM)

For every remaining edge $(i, j)$ in $G_{skel}$, we test directionality using LSTMs.
We define the history of variable $i$ up to time $t$ as $x_{i, <t}$.
To test if $j \to i$:

1.  **Restricted Model**: Predict $x_{i, t}$ using only its own history $x_{i, <t}$.
    $$ \mathcal{L}_{rest} = \sum_t (x_{i,t} - \text{LSTM}_R(x_{i, <t}))^2 $$
2.  **Unrestricted Model**: Predict $x_{i, t}$ using history of $i$ and $j$.
    $$ \mathcal{L}_{unrest} = \sum_t (x_{i,t} - \text{LSTM}_U(x_{i, <t}, x\_{j, <t}))^2 $$

We compute the F-statistic to quantify the improvement:
$$ F = \frac{(\mathcal{L}_{rest} - \mathcal{L}_{unrest}) / d*1}{\mathcal{L}*{unrest} / d_2} $$
If the p-value is significant, we assign a probability $P(j \to i) = 1 - \text{p-value}$.

### 3.3 Stage 3: Causal Graph Neural Network (GNN)

We treat the pairwise probabilities from Stage 2 as a noisy adjacency matrix $A_{noisy}$. A GNN refines this by aggregating global graph information.
**Node Features**: $H^{(0)} = I$ (One-hot identity).
**Graph Convolution**:
$$ H^{(l+1)} = \sigma \left( \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)} \right) $$
**Edge Prediction**:
$$ P(i \to j) = \sigma \left( \text{MLP}(H_i^{(L)} || H_j^{(L)}) \right) $$

**Hybrid Loss Function**:
$$ \mathcal{L} = \text{BCE}(P*{GNN}, A*{noisy}) + \lambda*1 ||P*{GNN}||\_1 $$
The L1 term ($\lambda_1=0.1$) encourages sparsity, implementing Occam's Razor.

### 3.4 Stage 4: Symbolic Logic Constraints

We enforce hard constraints to ensure the final graph is a valid DAG.

1.  **Sparsity**: $A_{final} = A_{GNN} \cdot \mathbb{I}(A_{GNN} > \tau)$ where $\tau=0.2$.
2.  **Acyclicity**: We detect cycles using Depth First Search (DFS). For any cycle $C$, we remove the weakest edge:
    $$ (u^_, v^_) = \text{argmin}_{(u,v) \in C} P(u \to v) $$
    $$ A_{final}[u^*, v^*] = 0 $$
    This iterative process guarantees a DAG structure.

---

## 4. Experiments

### 4.1 Dataset: Extended US Macroeconomics

We utilize the Statsmodels Macrodata (Reid, 2009), covering Q1 1959 to Q3 2009 ($T=203$).
**Variables (9)**:

1.  `realgdp`: Real Gross Domestic Product
2.  `realcons`: Real Personal Consumption Expenditures
3.  `realinv`: Real Gross Private Domestic Investment
4.  `realgovt`: Real Federal Consumption & Investment
5.  `realdpi`: Real Disposable Personal Income
6.  `cpi`: Consumer Price Index
7.  `m1`: M1 Money Stock
8.  `tbilrate`: 3-Month Treasury Bill Rate
9.  `unemp`: Unemployment Rate

**Preprocessing**:

- Level variables (GDP, Cons, Inv, Govt, Income, CPI, M1) are transformed via **Log-Difference** ($\Delta \ln x_t$) to represent growth rates.
- Rate variables (T-Bill, Unemp) are transformed via **First Difference** ($\Delta x_t$) to represent changes.
- All series are standardized to zero mean, unit variance.
- **Stationarity**: Confirmed via Augmented Dickey-Fuller (ADF) tests ($p < 0.05$).

### 4.2 Implementation Details

- **PC**: Significance $\alpha=0.05$.
- **LSTM**: Hidden dim=32, Layers=1, Max Lag=4 (1 year).
- **GNN**: Hidden dim=64, Layers=2, Dropout=0.2.
- **Optimizer**: Adam, LR=0.01.

### 4.3 Results & Analysis

#### 4.3.1 Discovered Causal Graph

The NSCD pipeline identified a sparse set of high-confidence edges. Notable relationships include:

1.  **Monetary Transmission Mechanism**:
    $$ \text{tbilrate} \to \text{realinv} \to \text{realgdp} $$

    - **Interpretation**: Changes in the interest rate (`tbilrate`) directly impact the cost of borrowing, which influences Investment (`realinv`). Investment is a key component of GDP, thus driving economic growth. This validates the standard Keynesian transmission channel.

2.  **Consumption Function**:
    $$ \text{realdpi} \to \text{realcons} $$

    - **Interpretation**: Real Disposable Income (`realdpi`) is the primary driver of Consumption (`realcons`). This aligns with the Permanent Income Hypothesis and basic consumption functions ($C = a + bY_d$).

3.  **Phillips Curve Dynamics**:
    $$ \text{unemp} \to \text{cpi} $$

    - **Interpretation**: The model detects a causal link from Unemployment to Inflation (`cpi`). This reflects the supply-side pressure on prices: lower unemployment leads to wage pressure, driving up inflation (the Phillips Curve).

4.  **Government Multiplier**:
    $$ \text{realgovt} \to \text{realgdp} $$
    - **Interpretation**: Government spending is identified as a direct driver of GDP, consistent with the definition of aggregate demand ($Y = C + I + G + NX$).

#### 4.3.2 Non-Linearity & Robustness

The Neural Granger stage was critical. Linear Granger tests often fail to detect the link between `tbilrate` and `realinv` due to the non-linear nature of investment decisions (e.g., "liquidity traps" or threshold effects). The LSTM successfully captured this. Furthermore, the Symbolic Acyclicity constraint removed feedback loops (e.g., $GDP \to Inv \to GDP$) that, while dynamically present, complicate causal interpretation in a static DAG framework.

---

## 5. Discussion

The results demonstrate that **NSCD** is not merely a "black box" predictor but a tool for **structural discovery**. By enforcing logical constraints, we produce graphs that are actionable for policymakers. For instance, the link `tbilrate -> realinv` suggests that the Federal Reserve can effectively control investment activity through interest rates, but the lack of a direct link `tbilrate -> realcons` suggests consumption is less sensitive to rate changes—a crucial insight for monetary policy targeting.

### 5.1 Limitations

- **Latent Confounders**: NSCD assumes causal sufficiency (no hidden common causes). In reality, unobserved variables (e.g., consumer sentiment, foreign shocks) could confound results.
- **Instantaneous Effects**: The current LSTM implementation focuses on lagged effects. Simultaneous interactions (within the same quarter) are handled by the PC skeleton but could be modeled more explicitly.

---

## 6. Conclusion & Future Work

We have introduced **Neuro-Symbolic Causal Discovery**, a robust framework for inferring causal structures from time-series data. By combining the non-linear power of Neural Networks with the interpretability of Symbolic Logic, NSCD recovers economically valid causal graphs from US Macroeconomic data.

**Future Work**:

1.  **Latent Variable Modeling**: Integrating Variational Autoencoders (VAEs) to infer hidden confounders.
2.  **Regime Switching**: Adapting the model to handle structural breaks (e.g., pre- vs. post-2008 crisis).
3.  **High-Dimensional Application**: Scaling the GNN to gene regulatory networks with thousands of variables.

---

## References

1.  Spirtes, P., Glymour, C., & Scheines, R. (2000). _Causality, Prediction, and Search_. MIT Press.
2.  Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. _Econometrica_.
3.  Runge, J., et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. _Science Advances_.
4.  Pamfil, R., et al. (2020). DYNOTEARS: Structure Learning from Time-Series Data. _AISTATS_.
5.  Tank, A., et al. (2018). Neural Granger Causality for Nonlinear Time Series. _arXiv_.
6.  Scarselli, F., et al. (2009). The Graph Neural Network Model. _IEEE Transactions on Neural Networks_.
7.  Reid, M. (2009). Statsmodels: Econometric and statistical modeling with python.
