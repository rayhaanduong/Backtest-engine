# Backtest-engine

This project explores short-term pattern searching in financial time series.
Inspired by Gupta et al. (2025), we extract patterns using a geometrical criterion:
patterns that precede one-sided market moves.

To evaluate the relevance of a pattern, we compute an information criterion based on Shannon entropy, which captures how strongly the pattern supports a directional view of the market.

By combining:

Geometric meaning (patterns that historically generated profits) and

Information-theoretic strength (confidence in directional bias),

we aim to isolate more “pure” patterns.
This is particularly useful in noisy, high-frequency (1 min bar) data, where raw pattern frequency can be misleading.

