import numpy as np
from sklearn.neighbors import KDTree
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock, cdist
from sklearn.preprocessing import StandardScaler

def get_patterns_labels_pnls(df, threshold = 15, window = 8, horizon = 2, pattern_only = False):
    """
    Extract patterns from data
    
    Args:
        df (DataFrame): DataFrame containing candlestick data (columns should be Open, High, Low, Close)
        start_list (ndarray) : list of indices to extract patterns
        threshold (double) : higher then threshold we buy, less than - threshold we sell
        window (int, optional): Number of candlesticks to include in the pattern. Defaults to 8
        horizon (int, optional): Number of periods to hold a position. Defaults to 2.
        pattern_only (bool, optional): Only returns patterns
    """
    
    ohlc = df[['Open','High','Low','Close']].values
    start_list = np.arange(len(df) - window - horizon)
    n_patterns = start_list.shape[0]
    
    indices = start_list[:, None] + np.arange(window)
    
    windows = ohlc[indices]  
    
   
    hl = windows[:,:,1] - windows[:,:,2]
    co = windows[:,:,3] - windows[:,:,0] 
    ho = windows[:,:,1] - windows[:,:,0]  
    ol = windows[:,:,0] - windows[:,:,2] 
    
   
    patterns = np.stack([hl , co, ho, ol], axis=2).reshape(n_patterns, -1)
   

    scaler = StandardScaler()
    patterns = scaler.fit_transform(patterns)
    patterns = scaler.transform(patterns)
        
    if (pattern_only):
        return patterns
    
    entry_index = start_list + window #First candlestick after the window
    exit_index  = entry_index + horizon #Exit after horizon
    
    opens = df["Open"].values   
    pnls = opens[exit_index] - opens[entry_index]  
    
    labels = np.where(pnls > threshold, "Buy", np.where(pnls < -threshold, "Sell", "Neutral"))    
    
    
    mask = labels != "Neutral"
    return patterns[mask], pnls[mask], labels[mask]



def H_global(labels):
    """
    Return the global entropy given labels

    Args:
        labels (ndarray): labels "Buy" or "Sell"
    """
    n = labels.size
    p_buy = labels[labels == "Buy"] if len(labels) > 0 else np.array([])
    p_sell = labels[labels == "Sell"] if len(labels) > 0 else np.array([])
    
    
    nb_buy = len(p_buy)
    nb_sell = n - nb_buy
    
    p_buy = nb_buy / n
    p_sell = nb_sell / n
    
    return -p_buy * np.log(p_buy) - p_sell * np.log(p_sell)


def get_neighborhood(patterns, theta):
    """
    Given a list of patterns, it returns the close neighborhood (without self) such as for every x,y in the neighborhood d(x,y) <= theta, where theta is the manhattan distance

    Args:
        patterns (ndarray): Array of patterns (size is N, Nb_features)
    """
    
    tree = KDTree(patterns, metric="manhattan")  
    neighbors = tree.query_radius(patterns, r=theta)
    
    return [nbrs[nbrs != i] for i, nbrs in enumerate(neighbors)]


def compute_local_entropy(label, indices):
    """
    It computes the local entropy (Shannon) of the patterns

    Args:
        label (ndarray): N patterns
        indices (ndarray): Neighborhood of the patterns

    Returns:
    
    H_local : (ndarray) : Local entropy (size N)
    
    """
    
    H_local = np.zeros(len(label))

    
    
    for i, neighbors in enumerate(indices):
        label_neighbors = label[neighbors]
        counts = Counter(label_neighbors)
        probs = np.array([counts.get("Buy", 0), counts.get("Sell", 0)]) 
    
        
        if probs.sum() > 0 : 
            probs = probs / probs.sum()
            probs = probs[probs>0]
            H_local[i] =  -(probs * np.log(probs)).sum()
        else :
            H_local[i] = 0
        
       

        
        
        
    return H_local

def IG(labels, patterns, theta):
    
    neighbors = get_neighborhood(patterns, theta)
    H_local = compute_local_entropy(labels, neighbors)
    H = H_global(labels)
    
    return H - H_local



def get_distances_pairwise(Buy_idx, Sell_idx, patterns, batch_size = 10000):
    
    """
    Calculate the distances L1 between the Buy and Sell patterns (pairwise)
    
    Args:
        Buys_idx (ndarray) : indices of Buy signal
        Sell_idx (ndarray) : indices of Sell signal
        patterns (ndarray) : Contains the features 
        batch_size (int , optional) : For low RAM computer. Defaults to 10000

    Returns:
        distances (ndarray) 
    """
    
    
    buy_patterns = patterns[Buy_idx]
    sell_patterns = patterns[Sell_idx]
    
    n_buy = buy_patterns.shape[0]
    
    distances = []
    
    for i in range(0, n_buy, batch_size):
        buy_batch = buy_patterns[i:i+batch_size]
        d = np.sum(np.abs(buy_batch[:, None, :] - sell_patterns[None, :, :]), axis=2)
        distances.append(d.ravel())

    distances = np.concatenate(distances)
    
    return distances


def match_patterns(patterns_test, B_prime, S_prime, patterns, rho, k):
    """
    Identify the patterns close enough to Buying and Selling patterns,
    with duplicate handling and Buy/Sell conflict resolution.

    Args:
        patterns_test (ndarray): test patterns
        B_prime (ndarray): indices corresponding to Buying signals
        S_prime (ndarray): indices corresponding to Selling signals
        patterns (ndarray): patterns from training dataset
        rho (float): threshold

    Returns:
        B_matches (list): indices of test patterns near a buying signal
        S_matches (list): indices of test patterns near a selling signal
    """


    
    patterns_B_prime = patterns[B_prime]
    patterns_S_prime = patterns[S_prime]

    # KDTree queries
    tree_B = KDTree(patterns_B_prime, metric="manhattan") if len(B_prime) > 0 else None
    tree_S = KDTree(patterns_S_prime, metric="manhattan") if len(S_prime) > 0 else None

    if tree_B:
        dist_B, _ = tree_B.query(patterns_test, k=k)  
        dist_B = dist_B.mean(axis=1)                  
    else:
        dist_B = np.full(len(patterns_test), np.inf)

    if tree_S:
        dist_S, _ = tree_S.query(patterns_test, k=k)
        dist_S = dist_S.mean(axis=1)
    else:
        dist_S = np.full(len(patterns_test), np.inf)
  
    
    
    # B_mask = (dist_B < rho) & ((dist_B < dist_S) | (dist_S >= rho))
    # S_mask = (dist_S < rho) & ((dist_S < dist_B) | (dist_B >= rho))
    
    score = dist_B / (dist_B + dist_S + 1e-12)
    B_mask = score < 0.45
    S_mask = score > 0.55

    B_matches = np.where(B_mask)[0].tolist()
    S_matches = np.where(S_mask)[0].tolist()

    return B_matches, S_matches




def plot_histograms_with_stats(distance_raw, distance_filtered, bins=100):
    """
    Plot 2 histograms showing the distance pairwise of Buy signals and Sell signals for the raw data and filtered data
    
    Args:
        distance_raw (ndarray): Raw data
        distance_filtered (ndarray): Filtered data
        bins (int, optional): Number of bins. Defaults to 100
    """
   
    mean_raw, median_raw = np.mean(distance_raw), np.median(distance_raw)
    mean_filtered, median_filtered = np.mean(distance_filtered), np.median(distance_filtered)

   
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  
    axes[0].hist(distance_raw, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(mean_raw, color='red', linestyle='dashed', linewidth=2, label=f'Moyenne={mean_raw:.2f}')
    axes[0].axvline(median_raw, color='green', linestyle='dashed', linewidth=2, label=f'Médiane={median_raw:.2f}')
    axes[0].legend()
    axes[0].set_title("Manhattan distances pairwise - raw")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Frequence")

  
    axes[1].hist(distance_filtered, bins=bins, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(mean_filtered, color='red', linestyle='dashed', linewidth=2, label=f'Moyenne={mean_filtered:.2f}')
    axes[1].axvline(median_filtered, color='green', linestyle='dashed', linewidth=2, label=f'Médiane={median_filtered:.2f}')
    axes[1].legend()
    axes[1].set_title("Manhattan distances pairwise- after filtration")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Frequence")

    plt.tight_layout()
    plt.show()



def construct_filtered_sets(labels, patterns, score, theta):

    """
    Construct filtered Buy and Sell sets

    Args :
        labels (np.array): array of "Buy" / "Sell"
        patterns (np.array) : features
        distances (np.array):  distances to neighbors
        indices (np.array): neighbor indices
        scores (np.array): array of pattern scores
        theta (float): distance threshold

    Returns:
            B_prime, S_prime: (lists) : list of indices for filtered Buy and Sell patterns
    """
    
    order = np.argsort(-score)
    patterns = np.array(patterns)
    B_idx, S_idx = [], []

    B_patterns = []
    S_patterns = []

    for idx in order:
        lbl = labels[idx]
        pt = patterns[idx]

        if lbl == "Buy":
            if S_patterns:
                dist = np.sum(np.abs(S_patterns - pt), axis=1)
                if np.min(dist) < theta:
                    continue
            B_idx.append(idx)
            B_patterns.append(pt)

        elif lbl == "Sell":
            if B_patterns:
                dist = np.sum(np.abs(B_patterns - pt), axis=1)
                if np.min(dist) < theta:
                    continue
            S_idx.append(idx)
            S_patterns.append(pt)

    return B_idx, S_idx