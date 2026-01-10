# BinSketch Algorithm


### Person 1: The Core (BinSketch & Inner Product)
**Responsibility:** Main sketching logic and Inner Product experiments.

#### 1. Implement BinSketch (The Mapping)
* Create a random mapping $i \to j$ where each dimension of the original vector $d$ maps to a bucket $N$.
* **Logic:** To sketch vector $\mathbf{a}$, for every bit set to $1$ at index $i$, find the mapped bucket $j$. Set $a_s[j] = 1$.
* **Crucial:** This is a **bitwise-OR** operation (not a count). If multiple $1$s map to the same bucket, the bucket value remains $1$.

#### 2. Implement Algorithm 1 (Inner Product Estimator)
To estimate the Inner Product (IP) between sketches $\mathbf{a}_s$ and $\mathbf{b}_s$, implement the formula from Algorithm 1:
1.  Define $n = 1 - 1/N$.
2.  Calculate raw intersection counts:
    * $n_{as} = |\mathbf{a}_s|$ (population count of sketch a)
    * $n_{bs} = |\mathbf{b}_s|$ (population count of sketch b)
    * $n_{as,bs} = \langle \mathbf{a}_s, \mathbf{b}_s \rangle$ (intersection count)
3.  Estimate original weights:
    * $n_a = \ln(1 - n_{as}/N) / \ln(n)$
    * (and similarly for $n_b$)
4.  **Final Estimate ($n_{a,b}$):**
    $$Est = n_a + n_b - \frac{1}{\ln(n)} \ln\left( n^{n_a} + n^{n_b} + \frac{n_{as,bs}}{N} - 1 \right)$$

#### 3. Implement Baseline (BCS)
* Implement the **Binary Compressed Sensing (BCS)** algorithm.
* Randomly assign indices to buckets, but calculate the **parity** (modulo 2 sum) of bits in each bucket rather than bitwise-OR.

#### 4. Run Experiment (Inner Product)
* **Metric:** Calculate Mean Square Error (MSE) between your estimate and the ground truth IP.
* **Ranking:** Set thresholds by finding the maximum existing Inner Product in the dataset first.

---

### Person 2: Jaccard Similarity (Sets)
**Responsibility:** Jaccard estimator and related experiments.
*Dependency:* You rely on Person 1's code for the IP estimates ($n_{a,b}$).

#### 1. Implement Algorithm 3 (Jaccard Estimator)
* Jaccard is calculated using the estimates from BinSketch.
* **Formula:**
    $$JS(a,b) = \frac{IP(a,b)}{Ham(a,b) + IP(a,b)}$$
* **Hamming Estimate (Algorithm 2):** Implement the Hamming estimator. The source specifies:
    $$Ham_{a,b} = n_a + n_b - 2n_{a,b}$$
    *(Note: Double-check your results against ground truth. If the source formula yields poor accuracy, verify the definition of Hamming for binary vectors, but start with the paper's Algorithm 3.)*

#### 2. Implement Baseline (MinHash)
* Implement standard **MinHash**.
* **Parameter:** Use $k$ permutations. The paper limits $k$ to **5,500 max**.

#### 3. Run Experiment (Jaccard)
* **Metric:** Calculate $-\ln(\text{MSE})$.
* **Ranking:** Partition data (90% train, 10% query). Calculate Precision, Recall, and F1 Score.
* **Thresholds:** Use Jaccard thresholds: $\{0.95, 0.9, 0.85, 0.8, 0.6, 0.5, 0.2, 0.1\}$.

---

### Person 3: Cosine Similarity & Efficiency
**Responsibility:** Angular distance and timing benchmarks.

#### 1. Implement Algorithm 4 (Cosine Estimator)
* Use the estimates from Person 1.
* **Formula:**
    $$Cos(a,b) = \frac{n_{a,b}}{\sqrt{n_a \cdot n_b}}$$
    *(This is the estimated IP divided by the geometric mean of the estimated weights).*

#### 2. Implement Baseline (SimHash)
* Implement **SimHash**: Generate a random vector $\mathbf{r} \in \{-1, +1\}^d$.
* Sketch is $1$ if $\langle \mathbf{u}, \mathbf{r} \rangle \ge 0$, else $0$.

#### 3. Run Experiment (Efficiency/Time)
* **Measure:** "Compression Time" — the time to generate hash functions + time to generate the sketch.
* **Comparison:** Measure time for **BinSketch vs. BCS vs. SimHash vs. MinHash**.
* **Plot:** Time (y-axis) vs. Compression Length $N$ (x-axis).
* **Note:** BinSketch and BCS should be very fast and overlapping near the bottom of the chart, while MinHash grows linearly.

---

## Phase 3: Evaluation (The Deliverables)

Combining the work of all three persons, you should produce the following graphs to match the source paper:

### 1. MSE Accuracy
* **X-Axis:** Compression Length ($N$). (Range: 100 to ~5500).
* **Y-Axis:** MSE (for IP) or $-\ln(\text{MSE})$ (for Jaccard/Cosine).

### 2. Ranking (F1 Score)
* **X-Axis:** Compression Length.
* **Y-Axis:** F1 Score.
* **Detail:** Generate these for different thresholds (e.g., 0.9, 0.5, 0.1) as shown in Figure 4.

### 3. Speedup
* A single plot showing **Compression Time vs. Length ($N$)** for the datasets.

> **Important Replication Detail:** The paper emphasizes that for BinSketch, you theoretically select $N$ based on sparsity $\psi$ and error probability $\delta$, but for the experiment, you should simply **vary $N$ manually** (e.g., 100, 500, 1000...) to observe the performance curve.