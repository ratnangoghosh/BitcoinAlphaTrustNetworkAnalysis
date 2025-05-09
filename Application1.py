import os
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import seaborn as sns

class DistrustCascadeSimulation:
    """
    Implementation of a signed-weighted distrust cascade simulation on Bitcoin trust network.
    
    We model distrust as a contagion where:
    - Each node is either Susceptible (S) or Infected (I) with distrust
    - Infection spreads through negative edges with probability proportional to:
        P(u→v) = α × f(u) × [-W(u,v)]₊
    - Where:
        - W(u,v) ∈ [-1,1] is the signed weight
        - [·]₊ takes only the negative part (so only distrust transmits)
        - f(u) is u's Fairness (reliability as a rater)
        - α ∈ (0,1) is a global infection-rate scalar
    """
    
    def __init__(self, csv_path, alpha=0.1, output_dir="/Users/navneetgupta/Downloads/NS"):
        """
        Initialize the simulation with data from a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing the trust network data
            alpha (float): Global infection rate scalar (between 0 and 1)
            output_dir (str): Directory where to save plot images
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.G = None  # Network graph
        self.fairness = {}  # Fairness scores for each node
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data(csv_path)
        
    def load_data(self, csv_path):
        """
        Load data from CSV and construct the trust network.
        
        Args:
            csv_path (str): Path to the CSV file with SOURCE, TARGET, RATING, TIME format
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, names=['SOURCE', 'TARGET', 'RATING', 'TIME'])
        df['SOURCE'] = df['SOURCE'].astype(str)
        df['TARGET'] = df['TARGET'].astype(str)
        self.G = nx.DiGraph()
        df['NORMALIZED_RATING'] = df['RATING'] / 10.0
        for _, row in df.iterrows():
            self.G.add_edge(row['SOURCE'], row['TARGET'], weight=row['NORMALIZED_RATING'])
        print(f"Graph constructed with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        self.calculate_fairness()
        
    def calculate_fairness(self):
        """
        Calculate fairness scores for each node.
        """
        print("Calculating fairness scores...")
        for node in self.G.nodes():
            outgoing_edges = list(self.G.out_edges(node, data=True))
            if outgoing_edges:
                ratings = [edge[2]['weight'] for edge in outgoing_edges]
                std_dev = np.std(ratings) if len(ratings) > 1 else 0
                self.fairness[node] = max(0.1, min(1.0, 1.0 / (1.0 + std_dev)))
            else:
                self.fairness[node] = 0.5

    def run_simulation(self, seed_nodes, max_iterations=100):
        """
        Run the distrust cascade simulation.
        """
        print(f"Starting simulation with {len(seed_nodes)} seed nodes and α={self.alpha}...")
        seed_nodes = set(str(node) for node in seed_nodes if str(node) in self.G.nodes())
        if not seed_nodes:
            raise ValueError("No valid seed nodes provided")
        infected = set(seed_nodes)
        susceptible = set(self.G.nodes()) - infected
        infection_history = {0: set(infected)}
        newly_infected_count = {0: len(seed_nodes)}
        for iteration in tqdm(range(1, max_iterations + 1)):
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
                print(f"Simulation converged after {iteration} iterations")
                break
            infected.update(newly_infected)
            susceptible -= newly_infected
            infection_history[iteration] = set(infected)
            newly_infected_count[iteration] = len(newly_infected)
        return {
            'final_infected': infected,
            'infection_size': len(infected),
            'infection_rate': len(infected) / self.G.number_of_nodes(),
            'iterations': len(infection_history) - 1,
            'infection_history': infection_history,
            'newly_infected_count': newly_infected_count
        }

    def identify_super_spreaders(self, top_k=10, sample_size=None):
        """
        Identify super-spreaders by running single-seed simulations.
        """
        print(f"Identifying top {top_k} super-spreaders...")
        nodes_to_test = list(self.G.nodes())
        if sample_size and sample_size < len(nodes_to_test):
            nodes_to_test = random.sample(nodes_to_test, sample_size)
        outbreak_sizes = {}
        for node in tqdm(nodes_to_test):
            results = self.run_simulation([node])
            outbreak_sizes[node] = results['infection_size']
        sorted_spreaders = sorted(outbreak_sizes.items(), key=lambda x: x[1], reverse=True)
        return {node: size for node, size in sorted_spreaders[:top_k]}

    def find_critical_threshold(self, seed_nodes, alpha_range=None, steps=10):
        """
        Find the critical threshold for alpha where distrust begins to cascade significantly.
        """
        print("Analyzing critical threshold...")
        if alpha_range is None:
            alpha_range = (0.01, 0.5)
        alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
        results = []
        original_alpha = self.alpha
        for alpha in tqdm(alphas):
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

    def test_defense_strategies(self, seed_nodes, strategies, budget=50):
        """
        Compare different defense strategies by simulating immunization.
        """
        print(f"Testing defense strategies with budget {budget}...")
        results = {}
        baseline = self.run_simulation(seed_nodes)
        results['baseline'] = baseline['infection_rate']
        for strategy in strategies:
            G_copy = self.G.copy()
            if strategy == 'top_fairness':
                sorted_fairness = sorted(self.fairness.items(), key=lambda x: x[1], reverse=True)
                nodes_to_remove = [node for node, _ in sorted_fairness[:min(budget, len(sorted_fairness))]]
                G_copy.remove_nodes_from(nodes_to_remove)
            elif strategy == 'negative_weight':
                negative_edges = [(u, v, data['weight']) for u, v, data in G_copy.edges(data=True) if data['weight'] < 0]
                sorted_edges = sorted(negative_edges, key=lambda x: x[2])
                edges_to_remove = [(u, v) for u, v, _ in sorted_edges[:min(budget, len(sorted_edges))]]
                G_copy.remove_edges_from(edges_to_remove)
            elif strategy == 'betweenness':
                betweenness = nx.betweenness_centrality(G_copy)
                sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
                nodes_to_remove = [node for node, _ in sorted_betweenness[:min(budget, len(sorted_betweenness))]]
                G_copy.remove_nodes_from(nodes_to_remove)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            original_G = self.G
            self.G = G_copy
            sim_results = self.run_simulation(seed_nodes)
            results[strategy] = sim_results['infection_rate']
            self.G = original_G
        return results

    def visualize_simulation(self, results):
        """
        Visualize the simulation results.
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(18, 12))
        # Growth plot
        plt.subplot(2, 2, 1)
        infections = [len(inf) for inf in results['infection_history'].values()]
        plt.plot(infections, linewidth=2.5, marker='o')
        plt.title('Distrust Infection Growth Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Infected Nodes')
        # New infections bar
        plt.subplot(2, 2, 2)
        new_inf = list(results['newly_infected_count'].values())
        plt.bar(range(len(new_inf)), new_inf, alpha=0.7)
        plt.title('New Infections Per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Newly Infected')
        # Infection rate
        plt.subplot(2, 2, 3)
        rate = [len(inf)/self.G.number_of_nodes() for inf in results['infection_history'].values()]
        plt.plot(rate, linewidth=2.5, marker='o')
        plt.axhline(0.5, linestyle='--', label='50% Threshold')
        plt.title('Infection Rate')
        plt.xlabel('Iteration')
        plt.ylabel('Fraction Infected')
        plt.legend()
        # Network sample
        plt.subplot(2, 2, 4)
        sub_nodes = list(results['infection_history'][0]) + random.sample(
            [n for n in results['final_infected'] if n not in results['infection_history'][0]],
            min(100, len(results['final_infected'])-len(results['infection_history'][0]))
        )
        subG = self.G.subgraph(sub_nodes)
        pos = nx.spring_layout(subG, seed=42)
        nx.draw(subG, pos, node_size=50, node_color='red', with_labels=False)
        plt.title('Sample of Infected Nodes')
        # save
        out = os.path.join(self.output_dir, 'simulation_results.png')
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[saved] {out}")

    def visualize_critical_threshold(self, threshold_results):
        """
        Visualize the critical threshold analysis.
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 8))
        alphas = threshold_results['alphas']
        rates = threshold_results['infection_rates']
        changes = [rates[i+1]-rates[i] for i in range(len(rates)-1)] if len(rates)>1 else [0]
        crit_idx = changes.index(max(changes)) if changes else 0
        crit_alpha = alphas[crit_idx]
        plt.plot(alphas, rates, 'o-')
        plt.axvline(crit_alpha, linestyle='--', label=f'Critical α≈{crit_alpha:.3f}')
        plt.title('Critical Threshold Analysis')
        plt.xlabel('α')
        plt.ylabel('Final Infection Rate')
        plt.legend()
        out = os.path.join(self.output_dir, 'critical_threshold.png')
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[saved] {out}")

    def visualize_defense_strategies(self, defense_results):
        """
        Visualize the comparison of defense strategies.
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 8))
        strategies, rates = zip(*sorted(defense_results.items(), key=lambda x: x[1]))
        bars = plt.bar(strategies, rates)
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, h+0.01, f'{h:.2f}', ha='center')
        plt.title('Defense Strategy Effectiveness')
        plt.xlabel('Strategy')
        plt.ylabel('Final Infection Rate')
        out = os.path.join(self.output_dir, 'defense_strategies.png')
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[saved] {out}")

    def visualize_network_structure(self, sample_size=1000):
        """
        Visualize the network structure focusing on trust and distrust patterns.
        """
        print("Generating network structure visualization...")
        if self.G.number_of_nodes() > sample_size:
            degs = dict(self.G.degree())
            top = sorted(degs.items(), key=lambda x: x[1], reverse=True)[:sample_size//2]
            others = [n for n in self.G.nodes() if n not in dict(top)]
            samp = random.sample(others, sample_size//2)
            subG = self.G.subgraph([n for n,_ in top] + samp)
        else:
            subG = self.G
        plt.figure(figsize=(20, 16))
        # trust vs distrust
        plt.subplot(2, 2, 1)
        pos = nx.spring_layout(subG, seed=42)
        nx.draw_networkx_nodes(subG, pos, node_size=30, alpha=0.6, node_color='lightblue')
        pos_e = [(u,v) for u,v,d in subG.edges(data=True) if d['weight']>0]
        neg_e = [(u,v) for u,v,d in subG.edges(data=True) if d['weight']<0]
        nx.draw_networkx_edges(subG, pos, edgelist=pos_e, edge_color='green', alpha=0.5)
        nx.draw_networkx_edges(subG, pos, edgelist=neg_e, edge_color='red', width=1)
        plt.title('Trust (green) vs Distrust (red)')
        plt.axis('off')
        # distributions
        plt.subplot(2, 2, 2)
        weights = [d['weight'] for _,_,d in self.G.edges(data=True)]
        sns.histplot(weights, bins=20, kde=True)
        plt.title('Edge Weight Distribution')
        plt.axvline(0, color='red', linestyle='--')
        plt.subplot(2, 2, 3)
        degs = [d for _,d in self.G.degree()]
        sns.histplot(degs, bins=30, log_scale=(False,True))
        plt.title('Degree Distribution (log)')
        plt.subplot(2, 2, 4)
        fairness_vals = list(self.fairness.values())
        sns.histplot(fairness_vals, bins=20, kde=True)
        plt.title('Fairness Score Distribution')
        out = os.path.join(self.output_dir, 'network_structure.png')
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[saved] {out}")

    def visualize_super_spreaders(self, super_spreaders, sample_size=100):
        """
        Visualize the super-spreaders in the network.
        """
        print("Generating super-spreader visualization...")
        plt.figure(figsize=(18, 8))
        plt.subplot(1, 2, 1)
        nodes = list(super_spreaders.keys())
        sizes = list(super_spreaders.values())
        perc = [s/self.G.number_of_nodes()*100 for s in sizes]
        bars = plt.bar(range(len(nodes)), perc)
        for i, bar in enumerate(bars):
            h = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, h+0.5, f'{sizes[i]} ({perc[i]:.1f}%)', ha='center')
        plt.title('Top Super-Spreaders')
        plt.xticks(range(len(nodes)), nodes, rotation=45)
        # network view
        plt.subplot(1, 2, 2)
        spread = set(nodes)
        neigh = set()
        for n in spread:
            neigh |= set(self.G.successors(n)) | set(self.G.predecessors(n))
        sub_nodes = list(spread | set(random.sample(neigh, min(len(neigh), sample_size-len(spread)))))
        subG = self.G.subgraph(sub_nodes)
        pos = nx.spring_layout(subG, seed=42)
        nx.draw_networkx_nodes(subG, pos, node_color='lightgray', node_size=30)
        nx.draw_networkx_nodes(subG, pos, nodelist=list(spread), node_color='red', node_size=200)
        nx.draw_networkx_edges(subG, pos, alpha=0.3)
        nx.draw_networkx_labels(subG, pos, labels={n:n for n in spread}, font_size=10)
        plt.title('Super-spreaders Context')
        out = os.path.join(self.output_dir, 'super_spreaders.png')
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[saved] {out}")


if __name__ == "__main__":
    csv_path = "/Users/navneetgupta/Downloads/NS/soc-sign-bitcoinalpha.csv"
    simulation = DistrustCascadeSimulation(csv_path, alpha=0.1, output_dir="/Users/navneetgupta/Downloads/NS")
    simulation.visualize_network_structure()
    negative_incoming = {node: sum(1 for u,v,d in simulation.G.in_edges(node, data=True) if d['weight']<0)
                         for node in simulation.G.nodes()}
    seed_nodes = [n for n,_ in sorted(negative_incoming.items(), key=lambda x: x[1], reverse=True)[:5]]
    print(f"Using seed nodes: {seed_nodes}")
    results = simulation.run_simulation(seed_nodes)
    print(f"Final infection size: {results['infection_size']} nodes")
    simulation.visualize_simulation(results)
    super_spreaders = simulation.identify_super_spreaders(top_k=5, sample_size=100)
    simulation.visualize_super_spreaders(super_spreaders)
    threshold_results = simulation.find_critical_threshold(seed_nodes, steps=10)
    simulation.visualize_critical_threshold(threshold_results)
    defense_results = simulation.test_defense_strategies(
        seed_nodes, strategies=['top_fairness','negative_weight','betweenness'], budget=20)
    print("Defense results:", defense_results)
    simulation.visualize_defense_strategies(defense_results)

    
