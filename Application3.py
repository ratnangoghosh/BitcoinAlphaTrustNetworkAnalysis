# Install PyTorch (check compatibility here: https://pytorch.org/get-started/previous-versions/)
!pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Install torch-scatter, torch-sparse, torch-cluster, and torch-spline-conv
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
!pip install -q torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
!pip install -q torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install torch-geometric
!pip install -q torch-geometric

# Install CPU versions if needed (optional, comment out if using GPU)
# !pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# !pip install -q torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# !pip install -q torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# !pip install -q torch-geometric

# Ensure NumPy is installed
!pip install numpy==1.24.0


import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime
from collections import defaultdict, Counter
import matplotlib.dates as mdates

# Define a class for Anomalous Trust Pattern Detection
class AnomalousTrustDetector:
    def __init__(self, filepath):
        """Initialize with path to the BitcoinAlpha dataset."""
        self.filepath = filepath
        self.data = None
        self.G = None
        self.nx_G = None
        self.node_features = None
        self.edge_index = None
        self.edge_attr = None
        self.gnn_model = None
        self.node_embeddings = None

    def load_data(self):
        """Load and preprocess the BitcoinAlpha dataset."""
        print("Loading data...")

        # Load CSV data
        try:
            self.df = pd.read_csv(self.filepath, names=['source', 'target', 'rating', 'time'])
            print(f"Loaded {len(self.df)} ratings from {self.filepath}")
        except Exception as e:
            print(f"Error loading file: {e}")
            try:
                # Try with read_file API if file wasn't loaded directly
                import json
                content = window.fs.readFile(self.filepath, {'encoding': 'utf8'})
                lines = content.strip().split('\n')
                data = []
                for line in lines:
                    values = line.split(',')
                    if len(values) == 4:
                        source, target, rating, time = values
                        data.append({
                            'source': int(source),
                            'target': int(target),
                            'rating': int(rating),
                            'time': int(time)
                        })
                self.df = pd.DataFrame(data)
                print(f"Loaded {len(self.df)} ratings using alternative method")
            except Exception as e2:
                print(f"Second error loading file: {e2}")
                return False

        # Basic statistics
        print(f"Dataset statistics:")
        print(f"- Ratings: {len(self.df)}")
        print(f"- Unique sources: {self.df['source'].nunique()}")
        print(f"- Unique targets: {self.df['target'].nunique()}")
        print(f"- All nodes: {len(set(self.df['source']).union(set(self.df['target'])))}")
        print(f"- Rating range: {self.df['rating'].min()} to {self.df['rating'].max()}")
        print(f"- Positive ratings: {(self.df['rating'] > 0).sum()} ({(self.df['rating'] > 0).mean()*100:.1f}%)")
        print(f"- Negative ratings: {(self.df['rating'] < 0).sum()} ({(self.df['rating'] < 0).mean()*100:.1f}%)")

        # Convert timestamps to datetime for later temporal analysis
        self.df['datetime'] = pd.to_datetime(self.df['time'], unit='s')

        # Create NetworkX graph
        self.create_graph()
        return True

    def create_graph(self):
        """Create NetworkX and PyTorch Geometric graphs from the data."""
        print("Creating graph...")
        # Create directed weighted graph
        self.nx_G = nx.DiGraph()

        # Add edges with weights (ratings)
        for _, row in self.df.iterrows():
            self.nx_G.add_edge(row['source'], row['target'],
                              weight=row['rating'],
                              time=row['time'])

        # Collect all unique nodes to ensure consistent node indexing
        all_nodes = sorted(list(self.nx_G.nodes()))
        node_idx = {node: i for i, node in enumerate(all_nodes)}

        # Create edge index and attributes for PyTorch Geometric
        edge_index = []
        edge_attr = []
        edge_time = []

        for u, v, data in self.nx_G.edges(data=True):
            edge_index.append([node_idx[u], node_idx[v]])
            edge_attr.append(data['weight'])
            edge_time.append(data['time'])

        # Convert to required format
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
        self.edge_time = torch.tensor(edge_time, dtype=torch.long)

        # Create basic node features
        self.num_nodes = len(all_nodes)
        print(f"Graph created with {self.num_nodes} nodes and {len(edge_attr)} edges")

        # Create node features
        self.compute_node_features(node_idx)

        # Create PyTorch geometric Data object
        self.data = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes
        )

    def compute_node_features(self, node_idx):
        """Compute node features for graph learning."""
        print("Computing node features...")

        # Initialize basic features for each node
        features = np.zeros((self.num_nodes, 6))

        # For each node, calculate:
        # 1. In-degree (number of ratings received)
        # 2. Out-degree (number of ratings given)
        # 3. Average rating received
        # 4. Average rating given
        # 5. Rating variance received
        # 6. Proportion of negative ratings received

        for node, idx in node_idx.items():
            # Ratings received (in-edges)
            in_edges = list(self.nx_G.in_edges(node, data=True))
            in_ratings = [e[2]['weight'] for e in in_edges] if in_edges else [0]

            # Ratings given (out-edges)
            out_edges = list(self.nx_G.out_edges(node, data=True))
            out_ratings = [e[2]['weight'] for e in out_edges] if out_edges else [0]

            # Compute features
            features[idx, 0] = len(in_edges)  # In-degree
            features[idx, 1] = len(out_edges)  # Out-degree
            features[idx, 2] = np.mean(in_ratings) if in_edges else 0  # Avg rating received
            features[idx, 3] = np.mean(out_ratings) if out_edges else 0  # Avg rating given
            features[idx, 4] = np.var(in_ratings) if len(in_ratings) > 1 else 0  # Rating variance
            features[idx, 5] = sum(1 for r in in_ratings if r < 0) / max(1, len(in_ratings))  # Proportion negative

        # Normalize features
        scaler = StandardScaler()
        self.node_features = torch.tensor(scaler.fit_transform(features), dtype=torch.float)

    def build_gnn_model(self, hidden_channels=64):
        """Build and train a Graph Neural Network model."""
        print("Building and training GNN model...")

        class GNN(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.conv3 = GCNConv(hidden_channels, out_channels)

            def forward(self, x, edge_index, edge_weight=None):
                x = self.conv1(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
                x = self.conv2(x, edge_index, edge_weight)
                x = F.relu(x)
                x = self.conv3(x, edge_index, edge_weight)
                return x

        # Initialize model
        self.gnn_model = GNN(
            in_channels=self.node_features.size(1),
            hidden_channels=hidden_channels,
            out_channels=32  # Dimension of node embeddings
        )

        # Create positive edge weights for GNN training
        edge_weight = torch.ones_like(self.edge_attr.view(-1))

        # Since we're doing unsupervised learning, we'll use the model to learn
        # node embeddings that capture the graph structure
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)

        # Simple training loop to learn embeddings
        self.gnn_model.train()
        for epoch in range(100):
            optimizer.zero_grad()

            # Forward pass
            node_embeddings = self.gnn_model(
                self.node_features,
                self.edge_index,
                edge_weight
            )

            # Compute a simple reconstruction loss
            # Here we try to predict edge weights using node embeddings
            src, dst = self.edge_index
            pred = (node_embeddings[src] * node_embeddings[dst]).sum(dim=1)

            # Target is normalized edge weight
            target = self.edge_attr.view(-1)
            target = (target - target.mean()) / (target.std() + 1e-8)

            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')

        # Generate final embeddings
        self.gnn_model.eval()
        with torch.no_grad():
            self.node_embeddings = self.gnn_model(
                self.node_features,
                self.edge_index,
                edge_weight
            ).detach().cpu().numpy()

        print("GNN training completed")

    def detect_anomalous_clusters(self, n_clusters=5):
        """Detect unusual trust clusters using embeddings."""
        print("\n1. Detecting anomalous clusters...")

        # Apply K-means clustering on node embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.node_embeddings)

        # Analyze trust patterns within each cluster
        cluster_stats = []
        for i in range(n_clusters):
            # Get node indices for this cluster
            cluster_nodes = np.where(clusters == i)[0]

            # Calculate average trust within and between clusters
            inner_trust = []
            outer_trust = []

            for src, tgt, weight in zip(self.edge_index[0], self.edge_index[1], self.edge_attr):
                src_idx, tgt_idx = src.item(), tgt.item()
                if src_idx in cluster_nodes and tgt_idx in cluster_nodes:
                    inner_trust.append(weight.item())
                elif src_idx in cluster_nodes or tgt_idx in cluster_nodes:
                    outer_trust.append(weight.item())

            # Calculate statistics
            cluster_stats.append({
                'cluster': i,
                'size': len(cluster_nodes),
                'inner_trust_mean': np.mean(inner_trust) if inner_trust else 0,
                'outer_trust_mean': np.mean(outer_trust) if outer_trust else 0,
                'inner_trust_std': np.std(inner_trust) if len(inner_trust) > 1 else 0,
                'outer_trust_std': np.std(outer_trust) if len(outer_trust) > 1 else 0,
                'trust_ratio': (np.mean(inner_trust) / np.mean(outer_trust)
                               if outer_trust and np.mean(outer_trust) != 0
                               else float('inf'))
            })

        # Convert to DataFrame for analysis
        cluster_df = pd.DataFrame(cluster_stats)
        print("\nCluster statistics:")
        print(cluster_df)

        # Identify anomalous clusters
        # anomalous = cluster_df[
        #     (cluster_df['trust_ratio'] > 1.5) |
        #     (cluster_df['inner_trust_mean'] > 5 & (cluster_df['inner_trust_mean'] > 2 * cluster_df['outer_trust_mean'])) |
        #     (cluster_df['inner_trust_std'] < 0.5 & cluster_df['size'] > 10)  # Low variance might indicate collusion
        # ]
        anomalous = cluster_df[
    (cluster_df['trust_ratio'] > 1.5) |
    ((cluster_df['inner_trust_mean'] > 5) & (cluster_df['inner_trust_mean'] > 2 * cluster_df['outer_trust_mean'])) |
    ((cluster_df['inner_trust_std'] < 0.5) & (cluster_df['size'] > 10))
]

        if not anomalous.empty:
            print("\nPotential anomalous clusters detected:")
            print(anomalous)

            # Visualize clusters with PCA
            pca = PCA(n_components=2)
            embedded = pca.fit_transform(self.node_embeddings)

            plt.figure(figsize=(10, 8))
            for i in range(n_clusters):
                mask = clusters == i
                plt.scatter(embedded[mask, 0], embedded[mask, 1], label=f'Cluster {i}', alpha=0.7)

            plt.title('Node Embeddings Clustered with K-means')
            plt.xlabel('PCA Dimension 1')
            plt.ylabel('PCA Dimension 2')
            plt.legend()
            plt.tight_layout()
            plt.savefig('bitcoin_clusters.png')
            plt.close()

            return anomalous['cluster'].tolist()
        else:
            print("No clearly anomalous clusters detected based on current criteria")
            return []

    def detect_temporal_anomalies(self):
        """Detect unusual temporal patterns in trust ratings."""
        print("\n2. Detecting temporal anomalies...")

        # Group ratings by day
        self.df['date'] = self.df['datetime'].dt.date
        daily_ratings = self.df.groupby('date')['rating'].agg(['mean', 'count'])

        # Look for days with unusual activity or rating patterns
        # Calculate rolling averages for comparison
        daily_ratings['mean_rolling'] = daily_ratings['mean'].rolling(window=7, min_periods=1).mean()
        daily_ratings['count_rolling'] = daily_ratings['count'].rolling(window=7, min_periods=1).mean()

        # Calculate deviations from rolling average
        daily_ratings['mean_z'] = (daily_ratings['mean'] - daily_ratings['mean_rolling']) / daily_ratings['mean'].std()
        daily_ratings['count_z'] = (daily_ratings['count'] - daily_ratings['count_rolling']) / daily_ratings['count'].std()

        # Identify anomalous days
        anomalous_days = daily_ratings[
            (daily_ratings['mean_z'].abs() > 2) |  # Rating significantly different
            (daily_ratings['count_z'] > 2)  # Unusually high activity
        ]

        if not anomalous_days.empty:
            print("\nAnomalous days detected:")
            print(anomalous_days)

            # Plot temporal patterns
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Plot mean ratings over time
            ax1.plot(daily_ratings.index, daily_ratings['mean'], label='Daily Mean Rating')
            ax1.plot(daily_ratings.index, daily_ratings['mean_rolling'], label='7-day Rolling Average')
            ax1.set_ylabel('Mean Rating')
            ax1.set_title('Mean Rating Over Time')
            ax1.legend()

            # Highlight anomalous days
            for date in anomalous_days.index:
                ax1.axvline(x=date, color='r', linestyle='--', alpha=0.3)

            # Plot rating count over time
            ax2.plot(daily_ratings.index, daily_ratings['count'], label='Daily Rating Count')
            ax2.plot(daily_ratings.index, daily_ratings['count_rolling'], label='7-day Rolling Average')
            ax2.set_ylabel('Rating Count')
            ax2.set_xlabel('Date')
            ax2.set_title('Rating Activity Over Time')
            ax2.legend()

            # Highlight anomalous days
            for date in anomalous_days.index:
                ax2.axvline(x=date, color='r', linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.savefig('bitcoin_temporal_anomalies.png')
            plt.close()

            # Further investigate users involved in anomalous days
            anomalous_user_activity = self.df[self.df['date'].isin(anomalous_days.index)]
            most_active_users = anomalous_user_activity['source'].value_counts().head(10)

            print("\nMost active users during anomalous days:")
            print(most_active_users)

            return anomalous_days
        else:
            print("No significant temporal anomalies detected")
            return None

    def detect_trust_asymmetry(self):
        """Detect unusual asymmetric trust relationships."""
        print("\n3. Detecting trust asymmetry patterns...")

        # Create a dictionary to store bidirectional trust ratings
        trust_pairs = defaultdict(lambda: {'forward': None, 'backward': None})

        # Populate with edge data
        for src, tgt, data in self.nx_G.edges(data=True):
            pair_key = tuple(sorted([src, tgt]))
            if src < tgt:
                trust_pairs[pair_key]['forward'] = data['weight']
            else:
                trust_pairs[pair_key]['backward'] = data['weight']

        # Calculate trust asymmetry for bidirectional relationships
        asymmetry_data = []
        for pair, ratings in trust_pairs.items():
            if ratings['forward'] is not None and ratings['backward'] is not None:
                asymmetry = abs(ratings['forward'] - ratings['backward'])
                asymmetry_data.append({
                    'node1': pair[0],
                    'node2': pair[1],
                    'rating1': ratings['forward'],
                    'rating2': ratings['backward'],
                    'asymmetry': asymmetry,
                    'extreme': (ratings['forward'] > 0 and ratings['backward'] < 0) or
                              (ratings['forward'] < 0 and ratings['backward'] > 0)
                })

        # Convert to DataFrame
        asymmetry_df = pd.DataFrame(asymmetry_data)

        if not asymmetry_df.empty:
            # Identify highly asymmetric relationships
            high_asymmetry = asymmetry_df[
                (asymmetry_df['asymmetry'] > 10) |  # Large difference in ratings
                asymmetry_df['extreme']  # Opposite sign ratings
            ].sort_values('asymmetry', ascending=False)

            print(f"\nFound {len(asymmetry_df)} bidirectional relationships")
            print(f"Identified {len(high_asymmetry)} highly asymmetric relationships")

            if not high_asymmetry.empty:
                print("\nTop asymmetric relationships:")
                print(high_asymmetry.head(10))

                # Plot distribution of asymmetry
                plt.figure(figsize=(10, 6))
                sns.histplot(asymmetry_df['asymmetry'], bins=20)
                plt.axvline(x=10, color='r', linestyle='--')
                plt.title('Distribution of Trust Asymmetry')
                plt.xlabel('Trust Asymmetry (|rating1 - rating2|)')
                plt.ylabel('Count')
                plt.savefig('bitcoin_trust_asymmetry.png')
                plt.close()

                return high_asymmetry
            else:
                print("No highly asymmetric relationships found")
                return None
        else:
            print("No bidirectional relationships found in the network")
            return None

    def detect_negative_subgraphs(self):
        """Detect subgraphs with unusually high concentrations of negative ratings."""
        print("\n4. Detecting negative trust subgraphs...")

        # Create a graph with only negative edges for analysis
        neg_edges = [(u, v) for u, v, d in self.nx_G.edges(data=True) if d['weight'] < 0]
        neg_G = nx.DiGraph()
        neg_G.add_edges_from(neg_edges)

        # Find connected components in the undirected negative graph
        undirected_neg_G = neg_G.to_undirected()
        components = list(nx.connected_components(undirected_neg_G))

        print(f"Found {len(components)} connected components in the negative trust network")

        # Analyze large negative components
        large_components = [comp for comp in components if len(comp) >= 3]

        if large_components:
            print(f"Found {len(large_components)} negative components with 3+ nodes")

            # Analyze each large component
            negative_clusters = []
            for i, comp in enumerate(large_components):
                # Extract the subgraph with all original edges between these nodes
                subgraph = self.nx_G.subgraph(comp)

                # Count positive and negative edges
                pos_edges = sum(1 for u, v, d in subgraph.edges(data=True) if d['weight'] > 0)
                neg_edges = sum(1 for u, v, d in subgraph.edges(data=True) if d['weight'] < 0)
                total_edges = pos_edges + neg_edges

                # Calculate negative edge ratio
                neg_ratio = neg_edges / total_edges if total_edges > 0 else 0

                negative_clusters.append({
                    'component_id': i,
                    'size': len(comp),
                    'positive_edges': pos_edges,
                    'negative_edges': neg_edges,
                    'total_edges': total_edges,
                    'negative_ratio': neg_ratio,
                    'nodes': list(comp)
                })

            # Convert to DataFrame
            neg_clusters_df = pd.DataFrame(negative_clusters)

            # Find highly negative clusters
            highly_negative = neg_clusters_df[
                (neg_clusters_df['negative_ratio'] > 0.5) &  # More negative than positive
                (neg_clusters_df['size'] >= 3)  # At least 3 nodes
            ].sort_values('negative_ratio', ascending=False)

            if not highly_negative.empty:
                print("\nHighly negative clusters:")
                print(highly_negative[['component_id', 'size', 'positive_edges',
                                      'negative_edges', 'negative_ratio']])

                # Visualize the largest negative cluster
                largest_neg_idx = highly_negative.iloc[0]['component_id']
                largest_neg_nodes = highly_negative.iloc[0]['nodes']
                largest_neg_graph = self.nx_G.subgraph(largest_neg_nodes)

                plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(largest_neg_graph)

                # Draw negative edges in red
                neg_edges = [(u, v) for u, v, d in largest_neg_graph.edges(data=True) if d['weight'] < 0]
                nx.draw_networkx_edges(largest_neg_graph, pos, edgelist=neg_edges,
                                      edge_color='red', alpha=0.7)

                # Draw positive edges in green
                pos_edges = [(u, v) for u, v, d in largest_neg_graph.edges(data=True) if d['weight'] > 0]
                nx.draw_networkx_edges(largest_neg_graph, pos, edgelist=pos_edges,
                                      edge_color='green', alpha=0.5)

                # Draw nodes
                nx.draw_networkx_nodes(largest_neg_graph, pos, node_size=100, alpha=0.8)
                nx.draw_networkx_labels(largest_neg_graph, pos, font_size=8)

                plt.title(f'Largest Negative Cluster (Component {largest_neg_idx})')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('bitcoin_negative_cluster.png')
                plt.close()

                return highly_negative
            else:
                print("No highly negative clusters found")
                return None
        else:
            print("No significant negative components found")
            return None

    def detect_sybil_attacks(self):
        """Detect potential Sybil attacks (fake identities)."""
        print("\n5. Detecting potential Sybil attacks...")

        # Features that might indicate Sybil nodes:
        # 1. Created in a short time window
        # 2. Similar rating patterns
        # 3. Primarily rate each other positively
        # 4. Low activity outside their cluster

        # Group users by creation time (first appearance in the network)
        first_appearances = self.df.groupby(['source'])['time'].min()
        first_appearances = pd.DataFrame(first_appearances).reset_index()
        first_appearances.columns = ['node', 'first_time']

        # Cluster users by first appearance time
        first_appearances['first_datetime'] = pd.to_datetime(first_appearances['first_time'], unit='s')

        # Create time windows for analysis
        first_appearances['time_window'] = pd.cut(
            first_appearances['first_time'],
            bins=20,  # Adjust based on data timespan
            labels=False
        )

        # Find windows with unusual number of new users
        window_counts = first_appearances['time_window'].value_counts().sort_index()
        window_stats = window_counts.describe()
        suspicious_windows = window_counts[window_counts > window_stats['75%'] + 1.5 * (window_stats['75%'] - window_stats['25%'])]

        potential_sybils = []

        if not suspicious_windows.empty:
            # Analyze each suspicious time window
            for window_id in suspicious_windows.index:
                # Get nodes created in this window
                window_nodes = first_appearances[first_appearances['time_window'] == window_id]['node'].tolist()

                if len(window_nodes) < 3:
                    continue

                # Extract subgraph of interactions among these nodes
                try:
                    window_subgraph = self.nx_G.subgraph(window_nodes)
                except Exception:
                    continue

                # Calculate metrics
                internal_edges = window_subgraph.number_of_edges()

                # Count ratings from these nodes to each other vs. outside
                internal_ratings = []
                external_ratings_out = []
                external_ratings_in = []

                for node in window_nodes:
                    # Outgoing ratings to other nodes in the window
                    for neighbor in self.nx_G.successors(node):
                        if neighbor in window_nodes:
                            internal_ratings.append(self.nx_G[node][neighbor]['weight'])
                        else:
                            external_ratings_out.append(self.nx_G[node][neighbor]['weight'])

                    # Incoming ratings from outside the window
                    for predecessor in self.nx_G.predecessors(node):
                        if predecessor not in window_nodes:
                            external_ratings_in.append(self.nx_G[predecessor][node]['weight'])

                # Calculate statistics
                internal_positive_ratio = sum(1 for r in internal_ratings if r > 0) / max(1, len(internal_ratings))
                external_out_positive_ratio = sum(1 for r in external_ratings_out if r > 0) / max(1, len(external_ratings_out))
                external_in_positive_ratio = sum(1 for r in external_ratings_in if r > 0) / max(1, len(external_ratings_in))

                # Get the time range of this window
                window_start = min(first_appearances[first_appearances['time_window'] == window_id]['first_datetime'])
                window_end = max(first_appearances[first_appearances['time_window'] == window_id]['first_datetime'])

                # Criteria for potential Sybil group
                is_suspicious = (
                    len(window_nodes) >= 5 and  # At least 5 nodes
                    internal_edges > len(window_nodes) and  # Sufficient internal connectivity
                    internal_positive_ratio > 0.8 and  # Mostly positive internal ratings
                    (internal_positive_ratio > external_in_positive_ratio + 0.2)  # Significantly more positive internal ratings
                )

                if is_suspicious:
                    potential_sybils.append({
                        'window_id': window_id,
                        'nodes': window_nodes,
                        'node_count': len(window_nodes),
                        'internal_edges': internal_edges,
                        'internal_positive_ratio': internal_positive_ratio,
                        'external_out_positive_ratio': external_out_positive_ratio,
                        'external_in_positive_ratio': external_in_positive_ratio,
                        'window_start': window_start,
                        'window_end': window_end,
                        'window_duration': (window_end - window_start).total_seconds() / 3600  # hours
                    })

        if potential_sybils:
            # Convert to DataFrame
            sybil_df = pd.DataFrame(potential_sybils)

            print("\nPotential Sybil groups detected:")
            print(sybil_df[['window_id', 'node_count', 'internal_edges',
                           'internal_positive_ratio', 'window_duration']])

            # Visualize the largest potential Sybil group
            if not sybil_df.empty:
                largest_idx = sybil_df['node_count'].idxmax()
                largest_sybil = sybil_df.loc[largest_idx]
                largest_sybil_nodes = largest_sybil['nodes']

                # Create subgraph
                sybil_subgraph = self.nx_G.subgraph(largest_sybil_nodes)

                plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(sybil_subgraph, seed=42)

                # Draw edges colored by weight
                edges = sybil_subgraph.edges(data=True)
                edge_colors = ['green' if d['weight'] > 0 else 'red' for _, _, d in edges]

                nx.draw_networkx(
                    sybil_subgraph,
                    pos=pos,
                    with_labels=True,
                    node_color='skyblue',
                    edge_color=edge_colors,
                    node_size=200,
                    font_size=10
                )

                plt.title(f'Potential Sybil Group (Window {largest_sybil["window_id"]})')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('bitcoin_sybil_group.png')
                plt.close()

                return sybil_df
            else:
                print("No potential Sybil groups meet the criteria")
                return None
        else:
            print("No suspicious user creation patterns detected")
            return None

    def find_manipulation_rings(self):
        """Detect potential manipulation rings where users artificially inflate each other's reputation."""
        print("\n6. Detecting manipulation rings...")

        # Look for densely connected positive trust components
        # Create a graph with only strong positive edges
        strong_pos_edges = [(u, v) for u, v, d in self.nx_G.edges(data=True) if d['weight'] >= 8]
        pos_G = nx.DiGraph()
        pos_G.add_edges_from(strong_pos_edges)

        # Convert to undirected for component analysis
        undirected_pos_G = pos_G.to_undirected()
        components = list(nx.connected_components(undirected_pos_G))

        print(f"Found {len(components)} connected components in the strong positive trust network")

        # Analyze large positive components
        large_components = [comp for comp in components if len(comp) >= 3]

        if large_components:
            print(f"Found {len(large_components)} strong positive components with 3+ nodes")

            # Analyze each large component
            manipulation_rings = []
            for i, comp in enumerate(large_components):
                # Get the induced subgraph from the full graph
                subgraph = self.nx_G.subgraph(comp)

                # Calculate key metrics
                n_nodes = len(comp)
                possible_edges = n_nodes * (n_nodes - 1)  # Directed graph
                actual_edges = subgraph.number_of_edges()
                density = actual_edges / possible_edges if possible_edges > 0 else 0

                # Calculate average rating within component
                avg_rating = np.mean([d['weight'] for _, _, d in subgraph.edges(data=True)])

                # Calculate reciprocity of high ratings
                reciprocal_pairs = 0
                for u in comp:
                    for v in comp:
                        if u != v and subgraph.has_edge(u, v) and subgraph.has_edge(v, u):
                            if subgraph[u][v]['weight'] >= 8 and subgraph[v][u]['weight'] >= 8:
                                reciprocal_pairs += 1

                reciprocity = reciprocal_pairs / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

                # Count external ratings
                external_ratings_received = []
                for node in comp:
                    for pred in self.nx_G.predecessors(node):
                        if pred not in comp:
                            external_ratings_received.append(self.nx_G[pred][node]['weight'])

                avg_external_rating = np.mean(external_ratings_received) if external_ratings_received else 0

                # Rating differential (internal vs external)
                rating_diff = avg_rating - avg_external_rating

                # Criteria for potential manipulation ring
                is_suspicious = (
                    density > 0.5 and  # Densely connected
                    avg_rating > 8 and  # Very high internal ratings
                    reciprocity > 0.3 and  # High reciprocity of ratings
                    rating_diff > 3 and  # Significantly higher internal ratings than external
                    len(comp) >= 3  # At least 3 nodes
                )

                if is_suspicious:
                    manipulation_rings.append({
                        'component_id': i,
                        'nodes': list(comp),
                        'size': n_nodes,
                        'density': density,
                        'avg_rating': avg_rating,
                        'reciprocity': reciprocity,
                        'avg_external_rating': avg_external_rating,
                        'rating_diff': rating_diff
                    })

            # Convert to DataFrame
            if manipulation_rings:
                ring_df = pd.DataFrame(manipulation_rings)

                print("\nPotential manipulation rings detected:")
                print(ring_df[['component_id', 'size', 'density', 'avg_rating',
                              'reciprocity', 'rating_diff']])

                # Visualize the largest potential manipulation ring
                if not ring_df.empty:
                    largest_idx = ring_df['size'].idxmax()
                    largest_ring = ring_df.loc[largest_idx]
                    largest_ring_nodes = largest_ring['nodes']

                    # Create subgraph
                    ring_subgraph = self.nx_G.subgraph(largest_ring_nodes)

                    plt.figure(figsize=(10, 8))
                    pos = nx.spring_layout(ring_subgraph, seed=42)

                    # Create edge labels for weights
                    edge_labels = {(u, v): f'{d["weight"]}' for u, v, d in ring_subgraph.edges(data=True)}

                    # Draw the graph
                    nx.draw_networkx(
                        ring_subgraph,
                        pos=pos,
                        with_labels=True,
                        node_color='lightgreen',
                        node_size=300,
                        font_size=10,
                        arrows=True
                    )

                    # Draw edge labels
                    nx.draw_networkx_edge_labels(
                        ring_subgraph,
                        pos=pos,
                        edge_labels=edge_labels,
                        font_size=8
                    )

                    plt.title(f'Potential Manipulation Ring (Component {largest_ring["component_id"]})')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig('bitcoin_manipulation_ring.png')
                    plt.close()

                    return ring_df
                else:
                    print("No potential manipulation rings meet the criteria")
                    return None
            else:
                print("No potential manipulation rings found")
                return None
        else:
            print("No large positive trust clusters found")
            return None

    def run_full_analysis(self):
        """Run the complete anomalous trust pattern detection analysis."""
        print("Starting BitcoinAlpha Anomalous Trust Pattern Detection Analysis...")

        # Load and prepare data
        if not self.load_data():
            print("Failed to load data. Analysis aborted.")
            return

        # Build and train GNN model
        self.build_gnn_model()

        # Run all detection methods
        results = {}

        # 1. Detect anomalous clusters
        results['anomalous_clusters'] = self.detect_anomalous_clusters()

        # 2. Detect temporal anomalies
        results['temporal_anomalies'] = self.detect_temporal_anomalies()

        # 3. Detect trust asymmetry
        results['trust_asymmetry'] = self.detect_trust_asymmetry()

        # 4. Detect negative subgraphs
        results['negative_subgraphs'] = self.detect_negative_subgraphs()

        # 5. Detect Sybil attacks
        results['sybil_attacks'] = self.detect_sybil_attacks()

        # 6. Find manipulation rings
        results['manipulation_rings'] = self.find_manipulation_rings()

        # Generate summary report
        print("\n==========================================")
        print("BITCOIN ALPHA ANOMALOUS TRUST PATTERN ANALYSIS SUMMARY")
        print("==========================================")

        # Summarize findings
        print("\nKey findings:")

        if results['anomalous_clusters']:
            print(f"- {len(results['anomalous_clusters'])} anomalous trust clusters detected")

        if results['temporal_anomalies'] is not None and not results['temporal_anomalies'].empty:
            print(f"- {len(results['temporal_anomalies'])} days with unusual rating patterns")

        if results['trust_asymmetry'] is not None and not results['trust_asymmetry'].empty:
            print(f"- {len(results['trust_asymmetry'])} highly asymmetric trust relationships")

        if results['negative_subgraphs'] is not None and not results['negative_subgraphs'].empty:
            print(f"- {len(results['negative_subgraphs'])} negative trust subgraphs")

        if results['sybil_attacks'] is not None and not results['sybil_attacks'].empty:
            print(f"- {len(results['sybil_attacks'])} potential Sybil attack groups")

        if results['manipulation_rings'] is not None and not results['manipulation_rings'].empty:
            print(f"- {len(results['manipulation_rings'])} potential reputation manipulation rings")

        print("\nAnalysis complete.")
        return results


# Main execution function
def run_bitcoin_alpha_analysis(filepath):
    """Run the full Bitcoin Alpha anomalous trust pattern detection analysis."""
    detector = AnomalousTrustDetector(filepath)
    results = detector.run_full_analysis()
    return results

# Execute if running as script
if __name__ == "__main__":
    # Path to the BitcoinAlpha dataset
    filepath = "/content/soc-sign-bitcoinalpha.csv"
    run_bitcoin_alpha_analysis(filepath)