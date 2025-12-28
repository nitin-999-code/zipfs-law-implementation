# 📊 Zipf's Law Analysis on Song Lyrics Dataset

🔎 **Project Overview**

This project validates Zipf's Law on a corpus of song lyrics (Rihanna). Zipf's Law predicts that word frequency is roughly inversely proportional to its rank in a frequency table. This repository contains the lyrics CSV and starter code & instructions to reproduce a rank-frequency (log-log) plot and estimate the Zipf exponent.

🧠 **Objective**

- **Analyze** the word frequency distribution in the lyrics corpus.
- **Plot** a log-log rank vs frequency (Zipf) plot.
- **Estimate** a power-law exponent and comment on deviations.
- **Provide** reproducible steps and next-step ideas (stopword filtering, per-song analysis, robust power-law fitting).

📁 **Dataset Description**

The dataset is provided in `Rihanna.csv` and includes the following columns:

- `Artist` – Name of the artist
- `Title` – Song title
- `Lyric` – Full song lyrics
- `Album` – Album name
- `Year` – Year of release
- `Date` – Date of release

Notes: the `Lyric` column contains raw text (repeats, punctuation, and inconsistent casing are present). Preprocessing is recommended before analysis.

📝 **Methodology**

1. Data Preprocessing
	- Remove punctuation and special characters
	- Convert all text to lowercase
	- Tokenize lyrics into words

2. Word Frequency Count
	- Count occurrences of each token across the whole corpus (or per-song if desired)

3. Rank-Frequency Distribution
	- Sort words by frequency and compute rank
	- Plot `rank` vs `frequency` on a log-log scale

4. Estimating a Power-Law
	- Fit a simple linear model to log(rank) vs log(frequency) for a quick exponent estimate
	- Optionally use a specialized library (e.g., `powerlaw`) for more robust parameter estimation and goodness-of-fit tests

5. Visualizations
	- Log-log Zipf plot (rank vs frequency)
	- Bar chart of most common words
	- Optional: per-song Zipf plots or album/year comparisons

📊 **Visualizations (expected)**

- ✅ Zipf's Law Log-Log Plot
- 📈 Bar chart of top N word frequencies

📌 **Key Insights (what to expect)**

- The top-ranked words will dominate counts — common words and refrains (e.g., "love", "yeah", "baby") are frequent.
- A long-tail appears as rank increases: most words occur once or a few times.
- If the plot is approximately linear in log-log space, the corpus follows Zipf-like behavior. The estimated exponent `s` is often near 1 but may differ due to dataset size and lyric repetition.
- Deviations in tails can reflect stylistic or genre-specific language usage (lyrics include repeated choruses which emphasize some tokens).

-----------------------------------------

## Quick reproducible recipe (`analyze_zipf.py`)

Save the following as `analyze_zipf.py` at the repository root and run it.

```python
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Rihanna.csv')
text = ' '.join(df['Lyric'].dropna().astype(str).tolist())
tokens = re.findall(r"\w+", text.lower())
freq = Counter(tokens)
freq_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
ranks = np.arange(1, len(freq_items)+1)
frequencies = np.array([f for _, f in freq_items])

plt.figure(figsize=(8,5))
plt.loglog(ranks, frequencies, marker='.', linestyle='none')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title("Zipf plot — Rihanna lyrics")
plt.grid(True, which='both', ls='--', lw=0.5)
plt.savefig('zipf_rihanna.png', dpi=150)
plt.show()

# Quick exponent estimate
mask = frequencies > 0
coef = np.polyfit(np.log(ranks[mask]), np.log(frequencies[mask]), 1)
exponent = -coef[0]
print(f"Estimated Zipf exponent (s) ≈ {exponent:.3f}")
```

## Run instructions (macOS / zsh)

```bash
cd /Users/nitinsahu/Desktop/zipfs-law-implementation
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib numpy
python analyze_zipf.py
```

## Next steps (I can do these for you)

- Add `analyze_zipf.py` to the repo and run it to produce `zipf_rihanna.png` and the exponent estimate.
- Add stopword removal or per-song analyses and include resulting plots.
- Fit a robust power-law with the `powerlaw` package and include a short results section.

---

If you'd like, I will now add `analyze_zipf.py`, run the analysis, save the plot, and push the results to `main`.
