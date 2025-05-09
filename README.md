# Bitcoin Alpha Trust Network Analysis - A Network Science Approach

This repository contains code to analyze and simulate distrust cascades and triadic sign‐pattern transitions on the Bitcoin-Alpha trust network.

## Dataset

We use the **Bitcoin-Alpha** “who-trusts-whom” network from SNAP.  
- **Nodes:** 3,783  
- **Edges:** 24,186  
- **Edge weights (RATING):** –10 (total distrust) to +10 (total trust)  
- **Fields:**  
  - `SOURCE` (int): user ID of the rater  
  - `TARGET` (int): user ID of the rated  
  - `RATING` (int): –10…+10  
  - `TIME` (int): Unix timestamp (seconds since 1970-01-01 UTC)  
- **Original data:** [snap.stanford.edu/data/soc-sign-bitcoin-alpha.html](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

## Files

- **simulation.py**  
  Simulates the spread of “distrust infection” over the network and outputs an evolution animation (`evolution.mp4`).

- **triad_analysis.py**  
  Loads the data, partitions it into time windows, builds signed‐triad snapshots, computes transition matrices of triad sign-patterns, identifies “healing” vs “decay” transitions, and visualizes the transition heatmap.

## Usage

1. Place the CSV in the repo root as `soc-sign-bitcoinalpha.csv`.  
2. Install dependencies:
   ```bash
   pip install networkx pandas numpy matplotlib seaborn ffmpeg-python

# Application 1 - Where Distrust Lives? How it Spreads? Who Drives it? When it Explodes? How can you Stop it?

## 3. Core Implementation Details

### 3.1 Class Structure and Initialization  
The `DistrustCascadeSimulation` class encapsulates the entire simulation framework with the following initialization process:

```python
def __init__(self, csv_path, alpha=0.1, output_dir="/Users/navneetgupta/Downloads/NS"):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    self.alpha = alpha
    self.G = None              # Network graph
    self.fairness = {}         # Fairness scores for each node
    self.output_dir = output_dir
    os.makedirs(self.output_dir, exist_ok=True)
    self.load_data(csv_path)
````

Key parameters:

* `alpha`: Global infection‐rate scalar (0–1) controlling how quickly distrust spreads
* `csv_path`: Path to the input dataset
* `output_dir`: Directory for saving visualization outputs

### 3.2 Data Loading and Preprocessing

Loads data from the CSV and constructs a directed graph:

```python
def load_data(self, csv_path):
    df = pd.read_csv(csv_path, names=['SOURCE','TARGET','RATING','TIME'])
    df['SOURCE'] = df['SOURCE'].astype(str)
    df['TARGET'] = df['TARGET'].astype(str)
    self.G = nx.DiGraph()
    df['NORMALIZED_RATING'] = df['RATING'] / 10.0
    for _, row in df.iterrows():
        self.G.add_edge(row['SOURCE'], row['TARGET'], weight=row['NORMALIZED_RATING'])
    print(f"Graph constructed with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
    self.calculate_fairness()
```

Notable preprocessing steps:

1. Node IDs converted to strings
2. Ratings normalized to \[-1, 1]
3. Directed graph built with `weight` attributes
4. Fairness scores computed post‐construction

### 3.3 Fairness Calculation

Computes a “fairness” score for each node (inverse of rating variability):

```python
def calculate_fairness(self):
    for node in self.G.nodes():
        outgoing = list(self.G.out_edges(node, data=True))
        if outgoing:
            ratings = [d['weight'] for _,_,d in outgoing]
            std_dev = np.std(ratings) if len(ratings) > 1 else 0
            self.fairness[node] = max(0.1, min(1.0, 1.0/(1.0+std_dev)))
        else:
            self.fairness[node] = 0.5
```

* Low variability → high fairness
* High variability → low fairness
* Bounded in \[0.1, 1.0]; no outgoing edges → 0.5

---

## 4. Distrust Propagation Model

### 4.1 Conceptual Model

An SIR‐inspired epidemic model where:

* **S:** Susceptible nodes (not yet adopted distrust)
* **I:** Infected nodes (have adopted distrust and can spread it)
* **R:** (Not implemented—no recovery state)

### 4.2 Simulation Algorithm

Core algorithm in `run_simulation`:

```python
def run_simulation(self, seed_nodes, max_iterations=100):
    seed = set(str(n) for n in seed_nodes if str(n) in self.G)
    infected = set(seed)
    susceptible = set(self.G.nodes()) - infected
    history = {0: infected.copy()}
    for it in range(1, max_iterations+1):
        new_inf = set()
        for u in infected:
            for v in self.G.successors(u):
                if v not in infected and v not in new_inf:
                    w = self.G[u][v]['weight']
                    if w < 0:
                        p = self.alpha * self.fairness.get(u,0.5) * (-w)
                        if random.random() < p:
                            new_inf.add(v)
        if not new_inf:
            break
        infected |= new_inf
        susceptible -= new_inf
        history[it] = infected.copy()
    return {
        'final_infected': infected,
        'infection_size': len(infected),
        'infection_rate': len(infected)/self.G.number_of_nodes(),
        'iterations': len(history)-1,
        'infection_history': history
    }
```

Steps per iteration:

1. For each infected node, check outgoing edges
2. Compute infection probability: `α × fairness(u) × (-weight)`
3. Infect neighbors probabilistically
4. Update sets and track history until convergence

---

## 5. Analysis Capabilities

### 5.1 Super-Spreader Identification

```python
def identify_super_spreaders(self, top_k=10, sample_size=None):
    # run_simulation for each node (or sample) and rank by outbreak size
```

### 5.2 Critical Threshold Analysis

```python
def find_critical_threshold(self, seed_nodes, alpha_range=(0.01,0.5), steps=10):
    # sweep α values, run simulations, record infection rates
```

### 5.3 Defense Strategy Evaluation

```python
def test_defense_strategies(self, seed_nodes, strategies, budget=50):
    # baseline infection_rate
    # for each strategy: copy G, remove nodes/edges, rerun simulation
```

Strategies:

1. **Top Fairness:** remove highest‐fairness nodes
2. **Negative Weight:** remove most negative edges
3. **Betweenness:** remove highest betweenness‐centrality nodes

---

## 6. Visualization Components

### 6.1 Simulation Results Visualization

* Growth of distrust over time
* New infections per iteration
* Infection rate percentage
* Sample network of infected nodes

### 6.2 Network Structure Visualization

* Trust vs. distrust edges
* Edge weight distribution
* Degree distribution
* Fairness score distribution

### 6.3 Critical Threshold Visualization

Plot infection rate vs. α, marking phase transition.

### 6.4 Defense Strategy Comparison

Bar charts comparing infection rates under each intervention.

### 6.5 Super-Spreader Visualization

Bar chart of outbreak sizes and network highlight of top spreaders.

---

## 7. Main Execution Flow

1. Initialize with `α=0.1` and load data
2. Visualize network structure
3. Select seed nodes (most negative incoming edges)
4. Run main simulation & visualize results
5. Identify & visualize super-spreaders
6. Analyze & visualize critical threshold
7. Test & visualize defense strategies

---

## 8. Technical Implementation Details

### 8.1 Dependencies

* NetworkX
* Pandas
* NumPy
* Matplotlib / Seaborn
* tqdm

### 8.2 Computational Considerations

* Worst‐case O(n × e)
* Sampling for super-spreaders
* Stochastic simulation

### 8.3 Code Efficiency Notes

* Precompute fairness
* Copy graph for interventions
* Track newly infected to avoid redundancy

---

## 9. Theoretical Framework and Implications

### 9.1 Epidemic Model Foundation

Adapts SIR to trust context by weighting infection with edge strength and rater reliability.

### 9.2 Network Dynamics

* Cascade behavior
* Critical thresholds
* Influence propagation
* Intervention efficacy

### 9.3 Trust System Implications

Insights on reputation system stability and defense planning.

---

## 10. Conclusions and Future Work

### 10.1 Key Insights

* Distrust cascades with thresholds
* Super-spreaders drive outbreaks
* Network structure constrains spread
* Quantitative defense evaluation

### 10.2 Potential Extensions

* Temporal dynamics (use `TIME`)
* Recovery mechanisms
* More strategies
* Real-time monitoring integration
* Multiplex networks

### 10.3 Limitations

* Stochastic variability
* Scalability on large graphs
* Static network assumption

---

## 11. Output

### 11.1 `simulation_results.png`

* Top‐left: cumulative infections over time
* Top‐right: new infections per iteration
* Bottom‐left: infection rate with 50% threshold line
* Bottom‐right: sample network of infected nodes

### 11.2 `super_spreaders.png`

* Bar chart of super-spreader outbreak percentages
* Network subgraph highlighting top spreaders

### 11.3 `critical_threshold.png`

Plot of infection rate vs. α with identified tipping point.

### 11.4 `defense_strategies.png`

Bar chart comparing baseline vs. intervention rates; shows betweenness removal as most effective.

### 11.5 `network_structure.png`

* Spring layout (trust vs. distrust)
* Edge weight histogram
* Degree distribution (log-scale)
* Fairness score distribution

---
