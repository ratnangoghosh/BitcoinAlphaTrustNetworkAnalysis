import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from community import best_partition  # python-louvain package
except ImportError:
    print("Warning: python-louvain package not found. Installing community_louvain...")
    # Try to install the package
    import subprocess
    subprocess.check_call(["pip", "install", "python-louvain"])
    from community import best_partition
import os
from tqdm import tqdm
from collections import Counter

class BitcoinTrustAnalysis:
    def __init__(self, filepath, output_dir="./output"):
        """
        Initialize the analysis with the trust network data.

        Parameters:
        - filepath: Path to the CSV file containing the trust network data
        - output_dir: Directory to save output files and visualizations
        """
        self.filepath = filepath
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load the data
        print("Loading Bitcoin Alpha trust network data...")
        self.data = pd.read_csv(filepath)

        # Check expected column format and rename if needed
        if 'source' in self.data.columns and 'target' in self.data.columns and 'rating' in self.data.columns:
            pass  # Column names are already as expected
        elif 'SOURCE' in self.data.columns and 'TARGET' in self.data.columns and 'RATING' in self.data.columns:
            self.data.columns = ['source', 'target', 'rating', 'time']
        else:
            # Assume the format is source, target, rating, [optional time]
            cols = ['source', 'target', 'rating']
            if self.data.shape[1] > 3:
                cols.append('time')
            self.data.columns = cols

        # Create a directed graph
        print("Creating directed weighted graph...")
        self.G = nx.DiGraph()

        # Add edges with weights
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Adding edges"):
            self.G.add_edge(row['source'], row['target'], weight=row['rating'])

        print(f"Graph created with {len(self.G.nodes())} nodes and {len(self.G.edges())} edges")

        # Initialize the user profiles dictionary
        self.user_profiles = {}

    def create_user_profiles(self):
        """
        Create multi-dimensional user profiles based on trust metrics.
        """
        print("Creating multi-dimensional user profiles...")

        # Calculate basic network metrics
        print("Calculating network centrality metrics (this may take a while)...")

        # Use k=100 for approximation in large networks
        betweenness_centrality = nx.betweenness_centrality(self.G, k=100)

        # Safely handle eigenvector centrality for disconnected graphs
        try:
            # First try using NetworkX's eigenvector_centrality_numpy
            eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G, weight='weight')
        except (nx.AmbiguousSolution, ValueError, np.linalg.LinAlgError) as e:
            print(f"Eigenvector centrality calculation error: {e}")
            print("Network is disconnected. Calculating eigenvector centrality for each component...")

            # Initialize with zeros
            eigenvector_centrality = {node: 0.0 for node in self.G.nodes()}

            # Find connected components in the undirected version of the graph
            undirected = nx.Graph(self.G)
            components = list(nx.connected_components(undirected))
            print(f"Found {len(components)} connected components")

            # Calculate eigenvector centrality separately for each component
            for i, component in enumerate(components):
                if len(component) > 1:  # Only calculate for components with at least 2 nodes
                    subgraph = self.G.subgraph(component)
                    try:
                        # Try power iteration method first
                        comp_centrality = nx.eigenvector_centrality(subgraph, max_iter=1000, weight='weight')
                        for node, value in comp_centrality.items():
                            eigenvector_centrality[node] = value
                    except nx.PowerIterationFailedConvergence:
                        # If that fails, use degree centrality as fallback
                        comp_centrality = nx.degree_centrality(subgraph)
                        for node, value in comp_centrality.items():
                            eigenvector_centrality[node] = value

        in_degree_centrality = nx.in_degree_centrality(self.G)
        out_degree_centrality = nx.out_degree_centrality(self.G)

        # Identify communities
        print("Detecting communities using Louvain method...")
        # Create an undirected graph with absolute weights for community detection
        undirected_G = nx.Graph()
        for u, v, data in self.G.edges(data=True):
            # Use absolute value of weights to avoid negative degree issues
            weight = abs(data.get('weight', 1.0))
            undirected_G.add_edge(u, v, weight=weight)

        try:
            communities = best_partition(undirected_G)
        except Exception as e:
            print(f"Error in community detection: {e}")
            print("Using a fallback method for community detection...")
            # Fallback to connected components as communities
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected_G)):
                for node in component:
                    communities[node] = i

        # Process each node to create user profiles
        print("Building user profiles...")
        for node in tqdm(self.G.nodes(), desc="Processing nodes"):
            # Get trust given and received
            out_edges = list(self.G.out_edges(node, data=True))
            in_edges = list(self.G.in_edges(node, data=True))

            # Calculate trust metrics
            trust_given = [e[2]['weight'] for e in out_edges]
            trust_received = [e[2]['weight'] for e in in_edges]

            # Trust distribution metrics
            positive_trust_given = [w for w in trust_given if w > 0]
            negative_trust_given = [w for w in trust_given if w < 0]
            positive_trust_received = [w for w in trust_received if w > 0]
            negative_trust_received = [w for w in trust_received if w < 0]

            # Calculate trust selectivity (variance in ratings given)
            trust_selectivity = np.var(trust_given) if trust_given else 0

            # Calculate trust given vs received ratio
            trust_given_sum = sum(trust_given) if trust_given else 0
            trust_received_sum = sum(trust_received) if trust_received else 0
            trust_ratio = trust_given_sum / trust_received_sum if trust_received_sum != 0 else 0

            # Store the user profile
            self.user_profiles[node] = {
                # Basic metrics
                'out_degree': len(out_edges),
                'in_degree': len(in_edges),
                'out_degree_centrality': out_degree_centrality.get(node, 0),
                'in_degree_centrality': in_degree_centrality.get(node, 0),

                # Trust metrics
                'avg_trust_given': np.mean(trust_given) if trust_given else 0,
                'avg_trust_received': np.mean(trust_received) if trust_received else 0,
                'total_trust_given': trust_given_sum,
                'total_trust_received': trust_received_sum,
                'trust_ratio': trust_ratio,

                # Trust distribution
                'positive_trust_given_count': len(positive_trust_given),
                'negative_trust_given_count': len(negative_trust_given),
                'positive_trust_received_count': len(positive_trust_received),
                'negative_trust_received_count': len(negative_trust_received),
                'positive_trust_given_avg': np.mean(positive_trust_given) if positive_trust_given else 0,
                'negative_trust_given_avg': np.mean(negative_trust_given) if negative_trust_given else 0,
                'trust_selectivity': trust_selectivity,

                # Network position
                'betweenness_centrality': betweenness_centrality.get(node, 0),
                'eigenvector_centrality': eigenvector_centrality.get(node, 0),
                'community': communities.get(node, -1)
            }

        # Convert to DataFrame for easier manipulation
        self.user_profiles_df = pd.DataFrame.from_dict(self.user_profiles, orient='index')

        # Save the user profiles
        self.user_profiles_df.to_csv(os.path.join(self.output_dir, 'user_profiles.csv'))

        print(f"User profiles created and saved to {os.path.join(self.output_dir, 'user_profiles.csv')}")
        return self.user_profiles_df

    def identify_user_segments(self, n_clusters=5, method='kmeans'):
        """
        Identify distinct user segments using clustering algorithms.

        Parameters:
        - n_clusters: Number of clusters for KMeans
        - method: Clustering method ('kmeans' or 'dbscan')

        Returns:
        - DataFrame with user profiles and cluster assignments
        """
        print(f"Identifying user segments using {method}...")

        # Ensure user profiles are created
        if not hasattr(self, 'user_profiles_df'):
            self.create_user_profiles()

        # Select features for clustering (excluding non-numeric or redundant features)
        features = [
            'out_degree', 'in_degree', 'avg_trust_given', 'avg_trust_received',
            'trust_ratio', 'positive_trust_given_count', 'negative_trust_given_count',
            'positive_trust_received_count', 'negative_trust_received_count',
            'trust_selectivity', 'betweenness_centrality', 'eigenvector_centrality'
        ]

        # Handle NaN values and infinity
        for feature in features:
            self.user_profiles_df[feature] = pd.to_numeric(self.user_profiles_df[feature], errors='coerce')
            self.user_profiles_df[feature].fillna(0, inplace=True)
            self.user_profiles_df[feature].replace([np.inf, -np.inf], 0, inplace=True)

        X = self.user_profiles_df[features]

        # Standardize the features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply dimensionality reduction for visualization
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Apply t-SNE for better cluster visualization
        print("Applying t-SNE for visualization...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)

        # Perform clustering
        if method == 'kmeans':
            print(f"Clustering with KMeans (n_clusters={n_clusters})...")
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = clustering.fit_predict(X_scaled)
        elif method == 'dbscan':
            print("Clustering with DBSCAN...")
            clustering = DBSCAN(eps=0.5, min_samples=5)
            clusters = clustering.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # Add cluster assignments to the user profiles
        self.user_profiles_df['cluster'] = clusters

        # Add PCA and t-SNE components for visualization
        self.user_profiles_df['pca_x'] = X_pca[:, 0]
        self.user_profiles_df['pca_y'] = X_pca[:, 1]
        self.user_profiles_df['tsne_x'] = X_tsne[:, 0]
        self.user_profiles_df['tsne_y'] = X_tsne[:, 1]

        # Save segmented user profiles
        segmented_profiles_path = os.path.join(self.output_dir, f'user_segments_{method}.csv')
        self.user_profiles_df.to_csv(segmented_profiles_path)

        print(f"User segments identified and saved to {segmented_profiles_path}")
        return self.user_profiles_df

    def analyze_user_segments(self):
        """
        Analyze the characteristics of each user segment.

        Returns:
        - DataFrame with segment profiles
        """
        print("Analyzing user segments...")

        if 'cluster' not in self.user_profiles_df.columns:
            raise ValueError("User segments have not been identified yet. Call identify_user_segments first.")

        # Get segment statistics
        segment_stats = []

        # Features to aggregate
        agg_features = [
            'out_degree', 'in_degree', 'avg_trust_given', 'avg_trust_received',
            'trust_ratio', 'positive_trust_given_count', 'negative_trust_given_count',
            'positive_trust_received_count', 'negative_trust_received_count',
            'betweenness_centrality', 'eigenvector_centrality', 'trust_selectivity'
        ]

        for cluster_id in sorted(self.user_profiles_df['cluster'].unique()):
            cluster_data = self.user_profiles_df[self.user_profiles_df['cluster'] == cluster_id]

            # Calculate cluster statistics
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'size_percentage': len(cluster_data) / len(self.user_profiles_df) * 100
            }

            # Add mean values for each feature
            for feature in agg_features:
                stats[f'{feature}_mean'] = cluster_data[feature].mean()

            segment_stats.append(stats)

        # Create segment profiles DataFrame
        self.segment_profiles = pd.DataFrame(segment_stats)

        # Save segment profiles
        segment_profiles_path = os.path.join(self.output_dir, 'segment_profiles.csv')
        self.segment_profiles.to_csv(segment_profiles_path)

        print(f"Segment profiles created and saved to {segment_profiles_path}")
        return self.segment_profiles

    def identify_bridge_users(self, percentile=95):
        """
        Identify bridge users who connect otherwise separate communities.

        Parameters:
        - percentile: Percentile threshold for considering a node as a bridge (e.g., 95th percentile)

        Returns:
        - DataFrame of bridge users
        """
        print("Identifying bridge users...")

        # Calculate betweenness centrality if not already done
        if not hasattr(self, 'user_profiles_df') or 'betweenness_centrality' not in self.user_profiles_df.columns:
            self.create_user_profiles()

        # Define bridges as nodes with high betweenness centrality
        threshold = self.user_profiles_df['betweenness_centrality'].quantile(percentile/100)
        bridge_users = self.user_profiles_df[self.user_profiles_df['betweenness_centrality'] >= threshold]

        # Sort by betweenness centrality in descending order
        bridge_users = bridge_users.sort_values('betweenness_centrality', ascending=False)

        # Save bridge users to CSV
        bridge_users_path = os.path.join(self.output_dir, 'bridge_users.csv')
        bridge_users.to_csv(bridge_users_path)

        print(f"Bridge users identified and saved to {bridge_users_path}")
        return bridge_users

    def analyze_risk_attitudes(self):
        """
        Analyze how different user segments approach risk in their trust relationships.

        Returns:
        - DataFrame with risk metrics by segment
        """
        print("Analyzing risk attitudes across user segments...")

        if 'cluster' not in self.user_profiles_df.columns:
            raise ValueError("User segments have not been identified yet. Call identify_user_segments first.")

        # Define risk metrics
        self.user_profiles_df['risk_taking'] = (
            self.user_profiles_df['positive_trust_given_count'] /
            (self.user_profiles_df['positive_trust_given_count'] +
             self.user_profiles_df['negative_trust_given_count'] + 1e-10)
        )

        self.user_profiles_df['trust_volatility'] = (
            self.user_profiles_df['trust_selectivity']
        )

        # Analyze risk metrics by segment
        risk_by_segment = self.user_profiles_df.groupby('cluster').agg({
            'risk_taking': ['mean', 'std', 'min', 'max', 'count'],
            'trust_volatility': ['mean', 'std', 'min', 'max']
        })

        # Flatten the multi-index columns
        risk_by_segment.columns = ['_'.join(col).strip() for col in risk_by_segment.columns.values]
        risk_by_segment = risk_by_segment.reset_index()

        # Save risk analysis
        risk_analysis_path = os.path.join(self.output_dir, 'risk_attitude_by_segment.csv')
        risk_by_segment.to_csv(risk_analysis_path)

        print(f"Risk attitude analysis saved to {risk_analysis_path}")
        return risk_by_segment

    def visualize_segments(self):
        """
        Visualize the user segments using t-SNE and PCA projections.

        Returns:
        - Dictionary of figure objects
        """
        print("Visualizing user segments...")

        if 'cluster' not in self.user_profiles_df.columns:
            raise ValueError("User segments have not been identified yet. Call identify_user_segments first.")

        figures = {}

        try:
            # Create t-SNE visualization
            plt.figure(figsize=(12, 10))
            # Check for NaN values in visualization coordinates
            tsne_df = self.user_profiles_df.copy()
            tsne_df['tsne_x'] = pd.to_numeric(tsne_df['tsne_x'], errors='coerce')
            tsne_df['tsne_y'] = pd.to_numeric(tsne_df['tsne_y'], errors='coerce')
            tsne_df = tsne_df.dropna(subset=['tsne_x', 'tsne_y'])

            if len(tsne_df) > 0:
                scatter = plt.scatter(
                    tsne_df['tsne_x'],
                    tsne_df['tsne_y'],
                    c=tsne_df['cluster'],
                    cmap='viridis',
                    alpha=0.7,
                    s=50
                )
                plt.colorbar(scatter, label='Cluster')
                plt.title('User Segments Visualization (t-SNE)')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.tight_layout()

                # Save the figure
                tsne_path = os.path.join(self.output_dir, 'user_segments_tsne.png')
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
                figures['tsne'] = plt.gcf()
            else:
                print("Warning: Not enough valid data points for t-SNE visualization")
        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")

        try:
            # Create PCA visualization
            plt.figure(figsize=(12, 10))
            # Check for NaN values in visualization coordinates
            pca_df = self.user_profiles_df.copy()
            pca_df['pca_x'] = pd.to_numeric(pca_df['pca_x'], errors='coerce')
            pca_df['pca_y'] = pd.to_numeric(pca_df['pca_y'], errors='coerce')
            pca_df = pca_df.dropna(subset=['pca_x', 'pca_y'])

            if len(pca_df) > 0:
                scatter = plt.scatter(
                    pca_df['pca_x'],
                    pca_df['pca_y'],
                    c=pca_df['cluster'],
                    cmap='viridis',
                    alpha=0.7,
                    s=50
                )
                plt.colorbar(scatter, label='Cluster')
                plt.title('User Segments Visualization (PCA)')
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.tight_layout()

                # Save the figure
                pca_path = os.path.join(self.output_dir, 'user_segments_pca.png')
                plt.savefig(pca_path, dpi=300, bbox_inches='tight')
                figures['pca'] = plt.gcf()
            else:
                print("Warning: Not enough valid data points for PCA visualization")
        except Exception as e:
            print(f"Error creating PCA visualization: {e}")

        print(f"Visualizations saved to {self.output_dir}")
        return figures

    def run_complete_analysis(self, n_clusters=5, method='kmeans', bridge_percentile=95):
        """
        Run the complete trust-based market segmentation and behavioral analysis.

        Parameters:
        - n_clusters: Number of clusters for segmentation
        - method: Clustering method ('kmeans' or 'dbscan')
        - bridge_percentile: Percentile threshold for bridge users

        Returns:
        - Dictionary with all analysis results
        """
        print("Running complete trust-based market segmentation and behavioral analysis...")

        # Create user profiles
        user_profiles = self.create_user_profiles()

        # Identify user segments
        user_segments = self.identify_user_segments(n_clusters=n_clusters, method=method)

        # Analyze user segments
        segment_profiles = self.analyze_user_segments()

        # Identify bridge users
        bridge_users = self.identify_bridge_users(percentile=bridge_percentile)

        # Analyze risk attitudes
        risk_attitudes = self.analyze_risk_attitudes()

        # Visualize segments
        visualizations = self.visualize_segments()

        print("Complete analysis finished. All results saved to output directory.")

        # Return results
        return {
            'user_profiles': user_profiles,
            'user_segments': user_segments,
            'segment_profiles': segment_profiles,
            'bridge_users': bridge_users,
            'risk_attitudes': risk_attitudes,
            'visualizations': visualizations
        }


def main():
    """
    Main function to run the analysis.
    """
    # Define the file path to your Bitcoin Alpha dataset
    data_path = "/content/soc-sign-bitcoinalpha.csv"  # Update this to your file path

    # Create the analyzer
    analyzer = BitcoinTrustAnalysis(data_path, output_dir="/content/bitcoin_trust_analysis")

    # Run the complete analysis
    results = analyzer.run_complete_analysis(n_clusters=5, method='kmeans', bridge_percentile=95)

    # Print key insights
    print("\nKey Insights:")
    print(f"Number of user segments identified: {len(results['segment_profiles'])}")
    print(f"Largest segment size: {results['segment_profiles']['size'].max()} users")
    print(f"Number of bridge users identified: {len(results['bridge_users'])}")

    # Display segment profiles summary
    print("\nSegment Profiles Summary:")
    print(results['segment_profiles'].to_string())

    print("\nAnalysis complete! All results saved to the output directory.")


if __name__ == "__main__":
    main()
