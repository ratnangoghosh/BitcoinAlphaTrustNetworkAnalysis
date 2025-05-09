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

