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

## Overview

The code in `NSProj-Task1.py` implements a distrust cascade simulation within the Bitcoin Alpha trust network, modeled as a signed-weighted directed graph. It explores how distrust spreads through negative trust relationships, identifies influential nodes (super-spreaders), determines critical thresholds for widespread propagation, and evaluates defense strategies to mitigate distrust. The simulation leverages a contagion model inspired by epidemic spreading (SIR model), adapted to a trust context, and is encapsulated in the `DistrustCascadeSimulation` class.

### Purpose
The primary goal is to:
- Model the dynamics of distrust propagation in a trust-based network.
- Analyze network phenomena such as cascade behavior, critical thresholds, and the role of influential nodes.
- Provide actionable insights for managing trust platforms like Bitcoin Alpha by identifying key nodes and effective interventions.

---

## Key Components and Functionality

### 1. **Class Structure: `DistrustCascadeSimulation`**
- **Initialization**: 
  - Takes a CSV file path (trust network data), an `alpha` parameter (global infection rate scalar, 0 < α < 1), and an output directory for visualizations.
  - Constructs a NetworkX directed graph (`self.G`) and initializes a fairness dictionary (`self.fairness`).
- **Purpose**: Serves as the central framework for loading data, running simulations, and generating analyses.

```python
def __init__(self, csv_path, alpha=0.1, output_dir="/Users/navneetgupta/Downloads/NS"):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    self.alpha = alpha
    self.G = None
    self.fairness = {}
    self.output_dir = output_dir
    os.makedirs(self.output_dir, exist_ok=True)
    self.load_data(csv_path)
```

---

### 2. **Data Loading and Preprocessing**
- **Method**: `load_data`
- **Functionality**:
  - Reads a CSV file (format: `SOURCE, TARGET, RATING, TIME`) containing trust ratings from -10 (distrust) to +10 (trust).
  - Normalizes ratings to [-1, 1] by dividing by 10.
  - Constructs a directed graph (`nx.DiGraph`) with nodes as users and edges as trust relationships.
  - Calls `calculate_fairness` to assign reliability scores to nodes.
- **Dataset**: Bitcoin Alpha trust network, where nodes are users and edges are signed-weighted trust ratings.

```python
def load_data(self, csv_path):
    df = pd.read_csv(csv_path, names=['SOURCE', 'TARGET', 'RATING', 'TIME'])
    df['SOURCE'] = df['SOURCE'].astype(str)
    df['TARGET'] = df['TARGET'].astype(str)
    self.G = nx.DiGraph()
    df['NORMALIZED_RATING'] = df['RATING'] / 10.0
    for _, row in df.iterrows():
        self.G.add_edge(row['SOURCE'], row['TARGET'], weight=row['NORMALIZED_RATING'])
    self.calculate_fairness()
```

- **Fairness Calculation**: 
  - Method: `calculate_fairness`
  - Assigns a fairness score (0.1 to 1.0) to each node based on the consistency of their outgoing ratings (inverse of standard deviation).
  - Nodes with no outgoing edges default to a fairness of 0.5.

```python
def calculate_fairness(self):
    for node in self.G.nodes():
        outgoing_edges = list(self.G.out_edges(node, data=True))
        if outgoing_edges:
            ratings = [edge[2]['weight'] for edge in outgoing_edges]
            std_dev = np.std(ratings) if len(ratings) > 1 else 0
            self.fairness[node] = max(0.1, min(1.0, 1.0 / (1.0 + std_dev)))
        else:
            self.fairness[node] = 0.5
```

---

### 3. **Distrust Propagation Model**
- **Method**: `run_simulation`
- **Model**: Adapted SIR epidemic model with two states:
  - **Susceptible (S)**: Nodes not yet infected with distrust.
  - **Infected (I)**: Nodes that have adopted distrust and can spread it via negative edges.
  - No "Recovered" state.
- **Mechanism**:
  - Distrust spreads through negative edges with probability `P(u→v) = α × f(u) × [-W(u,v)]`, where:
    - `α`: Global infection rate scalar.
    - `f(u)`: Fairness of the source node `u`.
    - `W(u,v)`: Normalized edge weight (only negative weights contribute).
  - Starts with seed nodes, iterates until no new infections occur or a maximum iteration limit (default 100) is reached.
- **Output**: Dictionary with infection size, rate, history, and newly infected counts per iteration.

```python
def run_simulation(self, seed_nodes, max_iterations=100):
    infected = set(str(node) for node in seed_nodes if str(node) in self.G.nodes())
    susceptible = set(self.G.nodes()) - infected
    infection_history = {0: set(infected)}
    newly_infected_count = {0: len(seed_nodes)}
    for iteration in range(1, max_iterations + 1):
        newly_infected = set()
        for u in infected:
            for v in self.G.successors(u):
                if v not in infected and v not in newly_infected:
                    w_uv = self.G[u][v]['weight']
                    if w_uv < 0:
                        p_infection = self.alpha * self.fairness.get(u, 0.5) * (-w_uv)
                        if random.random() < p_infection:
                            newly_infected.add(v)
        if not newly_infected:
            break
        infected.update(newly_infected)
        infection_history[iteration] = set(infected)
        newly_infected_count[iteration] = len(newly_infected)
    return {
        'final_infected': infected,
        'infection_size': len(infected),
        'infection_rate': len(infected) / self.G.number_of_nodes(),
        'infection_history': infection_history,
        'newly_infected_count': newly_infected_count
    }
```

---

### 4. **Super-Spreader Identification**
- **Method**: `identify_super_spreaders`
- **Functionality**:
  - Runs single-seed simulations for a sample of nodes (default: all nodes or a specified `sample_size`).
  - Measures the outbreak size (number of infected nodes) for each seed.
  - Returns the top `top_k` nodes causing the largest outbreaks.
- **Purpose**: Identifies nodes with disproportionate influence on distrust spread.

```python
def identify_super_spreaders(self, top_k=10, sample_size=None):
    nodes_to_test = list(self.G.nodes())
    if sample_size and sample_size < len(nodes_to_test):
        nodes_to_test = random.sample(nodes_to_test, sample_size)
    outbreak_sizes = {}
    for node in nodes_to_test:
        results = self.run_simulation([node])
        outbreak_sizes[node] = results['infection_size']
    sorted_spreaders = sorted(outbreak_sizes.items(), key=lambda x: x[1], reverse=True)
    return {node: size for node, size in sorted_spreaders[:top_k]}
```

---

### 5. **Critical Threshold Analysis**
- **Method**: `find_critical_threshold`
- **Functionality**:
  - Tests a range of `alpha` values (default: 0.01 to 0.5, 10 steps).
  - Runs simulations with fixed seed nodes for each `alpha`.
  - Tracks infection rates to identify the tipping point where distrust cascades significantly.
- **Purpose**: Determines the `alpha` value at which distrust propagation transitions from localized to widespread.

```python
def find_critical_threshold(self, seed_nodes, alpha_range=None, steps=10):
    if alpha_range is None:
        alpha_range = (0.01, 0.5)
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    results = []
    original_alpha = self.alpha
    for alpha in alphas:
        self.alpha = alpha
        sim_results = self.run_simulation(seed_nodes)
        results.append({
            'alpha': alpha,
            'infection_rate': sim_results['infection_rate'],
            'infection_size': sim_results['infection_size']
        })
    self.alpha = original_alpha
    return {
        'alphas': [r['alpha'] for r in results],
        'infection_rates': [r['infection_rate'] for r in results],
        'infection_sizes': [r['infection_size'] for r in results]
    }
```

---

### 6. **Defense Strategies**
- **Method**: `test_defense_strategies`
- **Functionality**:
  - Evaluates three strategies under a fixed budget (e.g., 20 nodes/edges):
    - **Top Fairness**: Removes nodes with the highest fairness scores.
    - **Negative Weight**: Removes the most negative edges.
    - **Betweenness**: Removes nodes with the highest betweenness centrality.
  - Compares the infection rate post-intervention against a baseline.
- **Purpose**: Assesses the effectiveness of interventions in limiting distrust spread.

```python
def test_defense_strategies(self, seed_nodes, strategies, budget=50):
    results = {}
    baseline = self.run_simulation(seed_nodes)
    results['baseline'] = baseline['infection_rate']
    for strategy in strategies:
        G_copy = self.G.copy()
        if strategy == 'top_fairness':
            sorted_fairness = sorted(self.fairness.items(), key=lambda x: x[1], reverse=True)
            nodes_to_remove = [node for node, _ in sorted_fairness[:budget]]
            G_copy.remove_nodes_from(nodes_to_remove)
        elif strategy == 'negative_weight':
            negative_edges = [(u, v, d['weight']) for u, v, d in G_copy.edges(data=True) if d['weight'] < 0]
            sorted_edges = sorted(negative_edges, key=lambda x: x[2])
            edges_to_remove = [(u, v) for u, v, _ in sorted_edges[:budget]]
            G_copy.remove_edges_from(edges_to_remove)
        elif strategy == 'betweenness':
            betweenness = nx.betweenness_centrality(G_copy)
            sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            nodes_to_remove = [node for node, _ in sorted_betweenness[:budget]]
            G_copy.remove_nodes_from(nodes_to_remove)
        self.G = G_copy
        sim_results = self.run_simulation(seed_nodes)
        results[strategy] = sim_results['infection_rate']
        self.G = original_G
    return results
```

---

### 7. **Visualizations**
The code provides several visualization methods to interpret results, saved as PNG files in the output directory:
- **`visualize_simulation`**: Plots infection growth, new infections per iteration, infection rate, and a subgraph of infected nodes.
- **`visualize_network_structure`**: Displays trust vs. distrust edges, edge weight distribution, degree distribution, and fairness score distribution.
- **`visualize_super_spreaders`**: Shows a bar chart of top super-spreaders and their network context.
- **`visualize_critical_threshold`**: Plots infection rate vs. `alpha` with the critical threshold marked.
- **`visualize_defense_strategies`**: Compares infection rates across defense strategies in a bar chart.

---

### 8. **Main Execution Flow**
- **Location**: `__main__` block
- **Workflow**:
  1. Initializes the simulation with `alpha=0.1` and the Bitcoin Alpha dataset.
  2. Visualizes network structure.
  3. Selects seed nodes (top 5 with most negative incoming edges).
  4. Runs the simulation and visualizes results.
  5. Identifies and visualizes super-spreaders (top 5 from 100 sampled nodes).
  6. Analyzes and visualizes the critical threshold.
  7. Tests and visualizes defense strategies (budget = 20).

```python
if __name__ == "__main__":
    csv_path = "/Users/navneetgupta/Downloads/NS/soc-sign-bitcoinalpha.csv"
    simulation = DistrustCascadeSimulation(csv_path, alpha=0.1)
    simulation.visualize_network_structure()
    negative_incoming = {node: sum(1 for _, _, d in simulation.G.in_edges(node, data=True) if d['weight'] < 0)
                         for node in simulation.G.nodes()}
    seed_nodes = [n for n, _ in sorted(negative_incoming.items(), key=lambda x: x[1], reverse=True)[:5]]
    results = simulation.run_simulation(seed_nodes)
    simulation.visualize_simulation(results)
    super_spreaders = simulation.identify_super_spreaders(top_k=5, sample_size=100)
    simulation.visualize_super_spreaders(super_spreaders)
    threshold_results = simulation.find_critical_threshold(seed_nodes, steps=10)
    simulation.visualize_critical_threshold(threshold_results)
    defense_results = simulation.test_defense_strategies(
        seed_nodes, strategies=['top_fairness', 'negative_weight', 'betweenness'], budget=20)
    simulation.visualize_defense_strategies(defense_results)
```

---

## Technical Details
- **Dependencies**: NetworkX, Pandas, NumPy, Matplotlib, Seaborn, tqdm.
- **Complexity**: O(n × e) in the worst case for simulation (n = nodes, e = edges).
- **Efficiency**: Uses pre-computed fairness scores, sampling for large networks, and graph copies for defense testing.

---

## Key Insights
- **Distrust Dynamics**: Spreads in bursts but remains contained (~4-5% of nodes at `alpha=0.1`), limited by the dominance of positive trust.
- **Super-Spreaders**: A few nodes cause large outbreaks, often acting as bridges.
- **Critical Threshold**: A sharp increase in infection rate occurs around `alpha ≈ 0.055`.
- **Defense**: Removing high-betweenness nodes is the most effective strategy.

---

## Implications
This simulation provides a robust tool for understanding distrust in trust-based networks, offering insights into monitoring influential nodes, tuning platform parameters (e.g., `alpha`), and designing structural interventions to enhance network resilience.

--- 

# Application 2 - Group Users into Behaviour based Categories.



# Application 3 - Find the Odd Ones Out (Anomalies)
