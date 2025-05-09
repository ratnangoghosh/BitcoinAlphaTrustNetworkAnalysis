import os
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import seaborn as sns
from matplotlib import animation

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

    def __init__(self, csv_path, alpha=0.1, output_dir="./plots"):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.G = None
        self.fairness = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data(csv_path)

    # ... (other methods unchanged) ...

    def animate_evolution(self, seed_nodes, max_iterations=50, filename="network_evolution.mp4", fps=5):
        """
        Create and save a video animation of network evolution over iterations.

        Each frame shows node infection state (red=infected, blue=susceptible),
        and underlying trust (green)/distrust (red) edges.
        """
        # run simulation but capture history
        results = self.run_simulation(seed_nodes, max_iterations=max_iterations)
        history = results['infection_history']

        fig, ax = plt.subplots(figsize=(8, 8))
        pos = nx.spring_layout(self.G, seed=42)

        def update(frame):
            ax.clear()
            infected = history.get(frame, history[max(history.keys())])
            # draw nodes
            node_colors = ['red' if n in infected else 'blue' for n in self.G.nodes()]
            nx.draw_networkx_nodes(self.G, pos, node_size=50, node_color=node_colors, ax=ax)
            # draw edges
            trust_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d['weight'] > 0]
            distrust_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d['weight'] < 0]
            nx.draw_networkx_edges(self.G, pos, edgelist=trust_edges, edge_color='green', alpha=0.2, ax=ax)
            nx.draw_networkx_edges(self.G, pos, edgelist=distrust_edges, edge_color='red', alpha=0.2, ax=ax)
            ax.set_title(f"Iteration {frame}: {len(infected)} infected")
            ax.axis('off')

        ani = animation.FuncAnimation(fig, update, frames=sorted(history.keys()), interval=1000/fps)
        out_path = os.path.join(self.output_dir, filename)
        ani.save(out_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"[saved animation] {out_path}")

# Usage example (at bottom of script)
if __name__ == "__main__":
    csv_path = "/Users/navneetgupta/Downloads/NS/soc-sign-bitcoinalpha.csv"
    sim = DistrustCascadeSimulation(csv_path, alpha=0.1, output_dir="/Users/navneetgupta/Downloads/NS")
    # pick some seed nodes
    seed_nodes = list(sim.G.nodes())[:5]
    # generate evolution video
    sim.animate_evolution(seed_nodes, max_iterations=30, filename="evolution.mp4", fps=5)