import numpy as np
import pandas as pd
from Extract_patterns import get_patterns_labels_pnls, construct_filtered_sets, match_patterns, IG


def get_rolling_windows(df, train_months=3, val_months=1, test_months=1):
    """
    Return rollings windows (train validation test)
    
    """
    dates = df.index
    start = dates.min()
    end = dates.max()
    windows = []

    current = start
    while True:
        train_start = current
        train_end  = train_start + pd.DateOffset(months=train_months)

        val_start = train_end
        val_end = val_start + pd.DateOffset(months=val_months)

        test_start = val_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        df_train = df[(df.index >= train_start) & (df.index < train_end)]
        df_val = df[(df.index >= val_start) & (df.index < val_end)]
        df_test = df[(df.index >= test_start) & (df.index < test_end)]

        if len(df_train) and len(df_val) and len(df_test):
            windows.append((df_train, df_val, df_test))


        current = test_start

    return windows



def backtest(all_trades, opens, close, bps=2, target=10, stoploss=10):
    """
    Run a backtest on trading signals.
    
    Args:
        all_trades (list of dicts): each dict has {'idx': int, 'is_buy': bool}
        opens (np.array): open prices from data
        close (np.array): close prices from data
        bps (float): transaction cost per roudn 
        target (float): profit target per trade
        stoploss (float): stop loss per trade
    
    Returns:
        trades (list of dicts): each trade with entry, exit, pnl, win
    """
    trades = []
    current_state = False
    entry_price = None
    entry_idx = None

    i = 0
    while i < len(all_trades):
        dic = all_trades[i]
        idx = dic["idx"]
        price_open = opens[idx]

        # Enter a buy
        if dic["is_buy"] and not current_state:
            current_state = True
            entry_price = price_open
            entry_idx = idx
            i += 1
            continue
        
        # Exit a buy
        elif not dic["is_buy"] and current_state:
            trade_exit_idx = idx
            for j in range(entry_idx + 1, idx):
                if opens[j] >= entry_price + target:
                    trade_exit_idx = j
                    break
                elif opens[j] <= entry_price - stoploss:
                    trade_exit_idx = j
                    break

            pnl = close[trade_exit_idx] - entry_price - bps * 1e-4 * (close[trade_exit_idx] + entry_price)
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": trade_exit_idx,
                "entry_price": entry_price,
                "exit_price": close[trade_exit_idx],
                "pnl": pnl,
                "win": pnl > 0
            })

            current_state = False
            entry_price = None
            entry_idx = None

            # Skip overlapping signals
            while i < len(all_trades) and all_trades[i]["idx"] <= trade_exit_idx:
                i += 1
            continue

        i += 1

    # Close last trade if still open
    if current_state:
        trade_exit_idx = len(opens) - 1
        pnl = close[trade_exit_idx] - entry_price - bps * 1e-4 * (close[trade_exit_idx] + entry_price)
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": trade_exit_idx,
            "entry_price": entry_price,
            "exit_price": close[trade_exit_idx],
            "pnl": pnl,
            "win": pnl > 0
        })

    return trades




def  evaluate_window(train, val, theta, diameter_neigh, rho, threshold, horizon_train, alpha, beta, target, stoploss, bps, val_months):
    
     # --- Train phase ---
    patterns, pnls, labels = get_patterns_labels_pnls(train, threshold=threshold, window=4, horizon=horizon_train)
    
    if len(patterns) == 0 :
        return -999
    
    
    IG_score = IG(labels, patterns, theta=diameter_neigh)

    # Decay for recency weighting
    time_weights = np.exp(-beta * np.arange(len(pnls))[::-1])
    time_weights /= time_weights.sum()

    normalized_pnl = np.abs(pnls) / (np.abs(pnls).max() + 1e-8)

    Score = (alpha * IG_score) + (1 - alpha) * normalized_pnl
    Score *= time_weights  

    B_prime, S_prime = construct_filtered_sets(labels, patterns, Score, theta)
    
    if len(B_prime) == 0 or len(S_prime) == 0:
        return -999

    # --- Validation phase ---
    pattern_val = get_patterns_labels_pnls(val, threshold=threshold, window=4, horizon=5, pattern_only=True)
    B_matches, S_matches = match_patterns(pattern_val, B_prime, S_prime, patterns, rho, k=1)

    # Build trades
    all_trades = []
    for idx in B_matches: all_trades.append({"idx": idx, "is_buy": True})
    for idx in S_matches: all_trades.append({"idx": idx, "is_buy": False})
    all_trades = sorted(all_trades, key=lambda x: x["idx"])

    opens, close = val["Open"].values, val["Close"].values
    trades = backtest(all_trades, opens, close, bps=bps, target=target, stoploss=stoploss)
    

    if len(trades) < 10: 
        return -999

    pnls = np.array([t["pnl"] for t in trades])
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) + 1e-8
    sharpe_period = mean_pnl / std_pnl
    
    months_per_year = 12
    sharpe_annualized = sharpe_period * np.sqrt(months_per_year / val_months)
    


    return sharpe_annualized

    
    