# Bitcoin Alpha Trust Network Analysis - A Network Science Approach

This repository contains code to analyze and simulate distrust cascades and triadic sign‐pattern transitions on the Bitcoin-Alpha trust network.

Detailed Overview - [Presentation.pdf](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/Presentation.pdf)

## Dataset

Dataset - [soc-sign-bitcoinalpha.csv](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/soc-sign-bitcoinalpha.csv)

We use the **Bitcoin-Alpha** “who-trusts-whom” network from SNAP.  
- **Nodes:** 3,783  
- **Edges:** 24,186  
- **Edge weights (RATING):** –10 (total distrust) to +10 (total trust)  
- **Fields:**  
  - `SOURCE` (int): user ID of the rater  
  - `TARGET` (int): user ID of the rated  
  - `RATING` (int): –10…+10  
  - `TIME` (int): Unix timestamp (seconds since 1970-01-01 UTC)  
- **Original Data:** [snap.stanford.edu/data/soc-sign-bitcoin-alpha.html](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)
- **Original Paper:** [Edge Weight Prediction in Weighted Signed Networks](https://cs.stanford.edu/~srijan/pubs/wsn-icdm16.pdf)

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

Code - [Application1.py](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/Application1.py)

Detailed Explanation - [Application1_Report.pdf](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/Application1_Report.pdf)

## Overview

The code in `Application1.py` implements a distrust cascade simulation within the Bitcoin Alpha trust network, modeled as a signed-weighted directed graph. It explores how distrust spreads through negative trust relationships, identifies influential nodes (super-spreaders), determines critical thresholds for widespread propagation, and evaluates defense strategies to mitigate distrust. The simulation leverages a contagion model inspired by epidemic spreading (SIR model), adapted to a trust context, and is encapsulated in the `DistrustCascadeSimulation` class.

## Purpose
The primary goal is to:
- Model the dynamics of distrust propagation in a trust-based network.
- Analyze network phenomena such as cascade behavior, critical thresholds, and the role of influential nodes.
- Provide actionable insights for managing trust platforms like Bitcoin Alpha by identifying key nodes and effective interventions.

## Implications
This simulation provides a robust tool for understanding distrust in trust-based networks, offering insights into monitoring influential nodes, tuning platform parameters (e.g., `alpha`), and designing structural interventions to enhance network resilience.

## Output

![critical threshold](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/output_images/critical_threshold.png?raw=true)
![defense strategies](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/output_images/defense_strategies.png?raw=true)
![network structure](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/output_images/network_structure.png?raw=true)
![super spreaders](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/output_images/super_spreaders.png?raw=true)
![simulation results](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/output_images/simulation_results.png?raw=true)

# Application 2 - Group Users into Behaviour based Categories.

Code - [Application2.py](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/Application2.py)

Detailed Explanation - [Application2_Report.pdf](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/Application2_Report.pdf)

## Overview 

`Application2.py` is a Python script designed to analyze the Bitcoin Alpha trust network, a peer-to-peer cryptocurrency platform where users assign trust ratings to one another. The code leverages network science and machine learning techniques to extract user profiles, segment users into behavioral archetypes, and analyze their trust and risk attitudes. The implementation is encapsulated in the `BitcoinTrustAnalysis` class, which processes the trust network data, performs clustering, and generates insights into user behaviors and network dynamics.

## Purpose
The primary objectives of the code are:
- **Profile Users**: Build multi-dimensional user profiles using trust metrics, centrality measures, and community affiliations.
- **Segment Users**: Identify distinct user segments or archetypes through clustering.
- **Analyze Segments**: Characterize each segment’s trust behavior, risk attitudes, and network roles.
- **Visualize Results**: Produce visualizations to illustrate user segment distributions and relationships.


## Implications
- **Security**: Skeptics can enhance threat detection; trusting users need education.
- **Trust Optimization**: Segment-specific trust weighting improves accuracy.
- **User Experience**: Tailored interfaces enhance engagement.

## Ouput

![](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/output_images/user_segments_pca.png?raw=true)
![](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/output_images/user_segments_tsne.png?raw=true)

# Application 3 - Find the Odd Ones Out (Anomalies)

Code - [Application3.py](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/Application3.py)

Detailed Explanation - [Application3_Report.pdf](https://github.com/ratnangoghosh/BitcoinAlphaTrustNetworkAnalysis/blob/main/Application3_Report.pdf)

## Overview 

`Application3.py` is a Python script designed to detect anomalous trust patterns within the Bitcoin Alpha trust network, a peer-to-peer cryptocurrency platform where users assign trust ratings to one another ranging from -10 (strong distrust) to +10 (strong trust). The script employs a combination of graph neural networks (GNNs), clustering techniques, and temporal analysis to identify suspicious behaviors such as manipulation rings, Sybil attacks, and unusual trust clusters. The core functionality is encapsulated in the `AnomalousTrustDetector` class, which processes the trust network data, learns node embeddings using a GNN, and applies various detection methods to uncover anomalies.

## Purpose

The primary goal of the code is to enhance the security and integrity of decentralized trust networks by identifying potentially malicious or manipulative behaviors. Specifically, it aims to:

- **Detect Anomalous Clusters**: Identify groups of users with unusually high internal trust compared to their external relationships, which might indicate manipulation or collusion.
- **Identify Temporal Anomalies**: Pinpoint days with statistically unusual rating patterns, potentially signaling coordinated actions or significant platform events.
- **Uncover Trust Asymmetry**: Highlight highly asymmetric trust relationships that could reflect disputes or retaliatory actions.
- **Detect Negative Subgraphs**: Find clusters dominated by distrust, indicating conflict zones within the network.
- **Spot Sybil Attacks**: Identify groups of users created in a short time window with suspicious internal rating patterns, suggestive of fake identities.
- **Find Manipulation Rings**: Detect densely connected groups with artificially inflated trust ratings, pointing to coordinated reputation boosting.

## Implications

This code provides a robust framework for analyzing trust networks, offering insights into vulnerabilities and manipulative behaviors. It supports platform integrity by identifying anomalies that could undermine trust mechanisms, making it a valuable tool for securing decentralized systems like Bitcoin Alpha.

# References

- [Dynamics of Epidemic Spreading on Signed Networks](https://www.sciencedirect.com/science/article/pii/S0960077921006482)
- [Unsupervised Clustering of Bitcoin Transactions](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-023-00525-y)
- [REV2 - Fraudulent User Prediction in Rating Platforms](https://dl.acm.org/doi/10.1145/3159652.3159729)
