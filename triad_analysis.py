import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import product
from datetime import datetime, timedelta

# Load the data
def load_data(file_path="/Users/navneetgupta/Downloads/NS/soc-sign-bitcoinalpha.csv"):
    """
    Load the Bitcoin Alpha trust network data from a CSV file or use provided data.
    Returns a DataFrame with columns: SOURCE, TARGET, RATING, TIME.
    """
    if file_path:
        try:
            df = pd.read_csv(file_path, sep=',')
            return df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    else:
        # Use the provided data as a fallback
        data = """
7188,1,10,1407470400
430,1,10,1376539200
3134,1,10,1369713600
3026,1,10,1350014400
3010,1,10,1347854400
804,1,10,1337572800
160,1,10,1394683200
95,1,9,1384578000
377,1,7,1414728000
888,1,7,1365652800
89,1,7,1351742400
1901,1,6,1411790400
161,1,6,1413345600
256,1,6,1342584000
351,1,5,1416373200
3329,1,5,1389934800
3341,1,5,1388984400
649,1,5,1384491600"""
        
        # Create a DataFrame from the string data
        lines = data.strip().split('\n')
        header = lines[0].split(',')
        data_rows = [line.split(',') for line in lines[1:]]
        
        df = pd.DataFrame(data_rows, columns=header)
        
        # Convert data types
        df['SOURCE'] = df['SOURCE'].astype(int)
        df['TARGET'] = df['TARGET'].astype(int)
        df['RATING'] = df['RATING'].astype(int)
        df['TIME'] = df['TIME'].astype(int)
        
        return df

def preprocess_data(df):
    """
    Preprocess the data:
    1. Convert timestamps to datetime
    2. Sort by timestamp
    3. Convert ratings to signed edges (+1 for positive, -1 for negative)
    """
    # Convert timestamp to datetime
    df['DATETIME'] = pd.to_datetime(df['TIME'], unit='s')
    
    # Sort by timestamp
    df = df.sort_values('TIME')
    
    # Convert ratings to signs (+1 for positive, -1 for negative)
    df['SIGN'] = np.where(df['RATING'] > 0, 1, -1)
    
    return df

def create_time_windows(df, window_size='W'):
    """
    Partition the data into time windows.
    
    Parameters:
    - df: DataFrame with trust data
    - window_size: pandas frequency string (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns a dictionary with time window labels as keys and DataFrames as values.
    """
    # Create time window labels
    df['WINDOW'] = df['DATETIME'].dt.to_period(window_size)
    
    # Group data by time windows
    windows = {str(window): group for window, group in df.groupby('WINDOW')}
    
    return windows

def build_snapshot_network(df):
    """
    Build a directed signed network from a DataFrame.
    
    Parameters:
    - df: DataFrame with trust data
    
    Returns a NetworkX DiGraph with signed edges.
    """
    G = nx.DiGraph()
    
    # Add all nodes from the dataframe
    all_nodes = set(df['SOURCE'].unique()) | set(df['TARGET'].unique())
    G.add_nodes_from(all_nodes)
    
    # Add edges with signs
    for _, row in df.iterrows():
        G.add_edge(row['SOURCE'], row['TARGET'], sign=row['SIGN'], rating=row['RATING'])
    
    return G

def extract_triads(G):
    """
    Extract all directed triads from the network with their sign patterns.
    
    Parameters:
    - G: NetworkX DiGraph with signed edges
    
    Returns a dictionary where keys are triad IDs (frozen sets of nodes) and 
    values are dictionaries of edge signs.
    """
    triads = {}
    
    # Get all nodes
    nodes = list(G.nodes())
    
    # Check all possible node triplets
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            for k in range(j+1, len(nodes)):
                # Create a subgraph with the three nodes
                node_set = [nodes[i], nodes[j], nodes[k]]
                
                # Get all possible directed edges among the three nodes
                edges = {}
                for u, v in product(node_set, node_set):
                    if u != v and G.has_edge(u, v):
                        edges[(u, v)] = G[u][v]['sign']
                
                # Only include triplets that form at least one triad 
                # (have at least 2 edges between them)
                if len(edges) >= 2:
                    triad_id = frozenset(node_set)
                    triads[triad_id] = edges
    
    return triads

def categorize_triad(edges):
    """
    Categorize a triad based on its edge sign pattern.
    
    For simplicity, we'll use a string representation of the triad pattern:
    +++ (all positive), ++- (two positive, one negative), etc.
    
    Parameters:
    - edges: Dictionary of directed edges with signs
    
    Returns a string representation of the triad pattern.
    """
    pos_count = sum(1 for sign in edges.values() if sign > 0)
    neg_count = sum(1 for sign in edges.values() if sign < 0)
    
    # Create a pattern string like "+++" or "++-" or "--+" etc.
    pattern = '+' * pos_count + '-' * neg_count
    
    return pattern

def build_transition_matrix(windows, window_labels):
    """
    Build a transition matrix for triad patterns across consecutive time windows.
    
    Parameters:
    - windows: Dictionary of network snapshots for each time window
    - window_labels: Ordered list of window labels
    
    Returns a transition count matrix and a transition probability matrix.
    """
    # All possible patterns (we'll discover these dynamically)
    patterns = set()
    
    # Store triads and their patterns for each window
    window_triads = {}
    
    # Extract triads for each window
    for window_label in window_labels:
        G = windows[window_label]
        triads = extract_triads(G)
        
        # Categorize each triad
        triad_patterns = {}
        for triad_id, edges in triads.items():
            pattern = categorize_triad(edges)
            triad_patterns[triad_id] = pattern
            patterns.add(pattern)
        
        window_triads[window_label] = triad_patterns
    
    # Convert to sorted list for consistent indexing
    patterns = sorted(patterns)
    pattern_to_idx = {pattern: idx for idx, pattern in enumerate(patterns)}
    
    # Initialize transition count and probability matrices
    n_patterns = len(patterns)
    transition_counts = np.zeros((n_patterns, n_patterns))
    
    # Count transitions
    for i in range(len(window_labels) - 1):
        current_window = window_labels[i]
        next_window = window_labels[i + 1]
        
        current_triads = window_triads[current_window]
        next_triads = window_triads[next_window]
        
        # Find triads that persist across windows
        common_triads = set(current_triads.keys()) & set(next_triads.keys())
        
        for triad_id in common_triads:
            from_pattern = current_triads[triad_id]
            to_pattern = next_triads[triad_id]
            
            from_idx = pattern_to_idx[from_pattern]
            to_idx = pattern_to_idx[to_pattern]
            
            transition_counts[from_idx, to_idx] += 1
    
    # Convert counts to probabilities
    transition_probs = np.zeros_like(transition_counts, dtype=float)
    row_sums = transition_counts.sum(axis=1)
    
    for i in range(n_patterns):
        if row_sums[i] > 0:
            transition_probs[i, :] = transition_counts[i, :] / row_sums[i]
    
    return transition_counts, transition_probs, patterns

def analyze_transitions(transition_counts, transition_probs, patterns):
    """
    Analyze the transition matrix to identify healing vs. decay transitions.
    
    Parameters:
    - transition_counts: Matrix of transition counts
    - transition_probs: Matrix of transition probabilities
    - patterns: List of triad patterns
    
    Returns dictionaries of healing and decay transitions with their probabilities.
    """
    # Classify patterns as balanced or unbalanced
    # In balance theory, balanced triads are:
    # - All positive relationships (+++)
    # - Two negative, one positive relationships (--+)
    balanced_patterns = [p for p in patterns if p.count('+') == len(p) or p.count('-') == 2]
    unbalanced_patterns = [p for p in patterns if p not in balanced_patterns]
    
    # Find healing transitions (unbalanced -> balanced)
    healing_transitions = {}
    for i, from_pattern in enumerate(patterns):
        if from_pattern in unbalanced_patterns:
            for j, to_pattern in enumerate(patterns):
                if to_pattern in balanced_patterns and transition_counts[i, j] > 0:
                    healing_transitions[(from_pattern, to_pattern)] = transition_probs[i, j]
    
    # Find decay transitions (balanced -> unbalanced)
    decay_transitions = {}
    for i, from_pattern in enumerate(patterns):
        if from_pattern in balanced_patterns:
            for j, to_pattern in enumerate(patterns):
                if to_pattern in unbalanced_patterns and transition_counts[i, j] > 0:
                    decay_transitions[(from_pattern, to_pattern)] = transition_probs[i, j]
    
    return healing_transitions, decay_transitions

def visualize_transition_matrix(transition_probs, patterns):
    """
    Visualize the transition probability matrix as a heatmap.
    
    Parameters:
    - transition_probs: Matrix of transition probabilities
    - patterns: List of triad patterns
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_probs, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=patterns, yticklabels=patterns)
    plt.title("Triad Pattern Transition Probabilities")
    plt.xlabel("To Pattern")
    plt.ylabel("From Pattern")
    plt.tight_layout()
    plt.savefig("triad_transition_heatmap.png")
    plt.show()

def main():
    """
    Main function to run the temporal signed-triad transition analysis.
    """
    print("Loading and preprocessing data...")
    df = load_data()
    
    if df is None or df.empty:
        print("Error: No data available. Exiting.")
        return
    
    # Display basic statistics
    print(f"Dataset loaded: {len(df)} edges, {len(set(df['SOURCE']) | set(df['TARGET']))} nodes")
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create time windows (weekly by default)
    print("Creating time windows...")
    window_size = 'W'  # Weekly
    windowed_data = create_time_windows(df, window_size)
    window_labels = sorted(windowed_data.keys())
    
    print(f"Created {len(window_labels)} time windows")
    
    # Build network snapshots for each window
    print("Building network snapshots...")
    network_snapshots = {}
    for window_label, window_df in windowed_data.items():
        network_snapshots[window_label] = build_snapshot_network(window_df)
    
    # Build transition matrix
    print("Building transition matrix...")
    transition_counts, transition_probs, patterns = build_transition_matrix(network_snapshots, window_labels)
    
    # Analyze transitions
    print("Analyzing transitions...")
    healing_transitions, decay_transitions = analyze_transitions(transition_counts, transition_probs, patterns)
    
    # Print results
    print("\n===== RESULTS =====")
    print(f"Found {len(patterns)} unique triad patterns across {len(window_labels)} time windows")
    
    print("\nTop Healing Transitions (Unbalanced -> Balanced):")
    for (from_pattern, to_pattern), prob in sorted(healing_transitions.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {from_pattern} -> {to_pattern}: {prob:.4f}")
    
    print("\nTop Decay Transitions (Balanced -> Unbalanced):")
    for (from_pattern, to_pattern), prob in sorted(decay_transitions.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {from_pattern} -> {to_pattern}: {prob:.4f}")
    
    # Visualize transition matrix
    print("\nVisualizing transition matrix...")
    visualize_transition_matrix(transition_probs, patterns)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()