**Project Overview**

- **Repository**: `zipfs-law-implementation`

- **Purpose**: This repository contains a dataset of Rihanna song lyrics (`Rihanna.csv`). The purpose of this project is to explore Zipf's Law on the lyrics — i.e., to show that word frequency vs rank follows an approximate power-law distribution (a straight line on a log-log plot).

**Dataset**
- **File**: `Rihanna.csv`
- **Columns**: index (auto), `Artist`, `Title`, `Album`, `Year`, `Date`, `Lyric`.
- **Notes**: The `Lyric` column includes raw song text (lowercase and punctuation present). This README assumes the CSV is UTF-8 and comma-separated.

**Zipf's Law (short)**
- Zipf's Law states that in many natural language corpora the frequency f of a word is approximately inversely proportional to its rank r: f(r) ~ C / r^s, where s is close to 1 for many languages. On a log-log plot of rank vs frequency this appears as an approximately straight line.

**Suggested analysis steps**
- 1) Tokenize all lyrics and normalize (lowercase, remove punctuation).
- 2) Count word frequencies across the corpus.
- 3) Sort words by frequency (rank) and plot frequency vs rank on a log-log scale.
- 4) Optionally fit a power-law (linear fit in log-log space) and report the slope (exponent s).

**Minimal Python recipe**
Save this as `analyze_zipf.py` and run in the repository root.

```python
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('Rihanna.csv')

# Join all lyrics
text = ' '.join(df['Lyric'].dropna().astype(str).tolist())

# Basic tokenization and normalization
tokens = re.findall(r"\w+", text.lower())

# Frequency counts
freq = Counter(tokens)

# Build rank-frequency list
freq_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
ranks = np.arange(1, len(freq_items)+1)
frequencies = np.array([f for _, f in freq_items])

# Log-log plot
plt.figure(figsize=(8,5))
plt.loglog(ranks, frequencies, marker='.', linestyle='none')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title("Zipf plot — Rihanna lyrics")
plt.grid(True, which='both', ls='--', lw=0.5)
plt.savefig('zipf_rhianna.png', dpi=150)
plt.show()

# Simple linear fit in log-log space (estimate exponent)
mask = frequencies > 0
log_r = np.log(ranks[mask])
log_f = np.log(frequencies[mask])
coef = np.polyfit(log_r, log_f, 1)
exponent = -coef[0]
print(f"Estimated Zipf exponent (s) ≈ {exponent:.3f}")

```

**Commands to run (macOS / zsh)**
- Create venv and install dependencies:

```bash
cd /Users/nitinsahu/Desktop/zipfs-law-implementation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib numpy
python analyze_zipf.py
```

**Interpretation**
- If Zipf's law holds roughly, the log-log plot will look approximately linear and the estimated exponent `s` will be near 1 (commonly between 0.8 and 1.2 for many corpora). Deviations can indicate dataset size, preprocessing choices (stopwords, stemming), or domain-specific word use (song lyrics often repeat choruses and refrains, which affects counts).

**Next steps / ideas**
- Remove common stopwords (e.g., using NLTK or a custom list) and re-run the analysis to compare.
- Analyze per-song Zipf distributions and compare exponents across albums/years.
- Fit a proper power-law using the `powerlaw` Python package for more robust estimates.

**Acknowledgements & License**
- This repository is a lightweight analysis starter. Check the licensing of the lyrics source before publishing derived datasets or visualizations publicly.

---

If you want, I can: create the `analyze_zipf.py` script in this repo, run the analysis and produce the plot, and then commit & push the changes to `main`.
