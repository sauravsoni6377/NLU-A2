# CSL 7640: Natural Language Understanding — Assignment 2

**Saurav Soni | B22AI035 | IIT Jodhpur**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

## Problem 1: Word Embeddings from IIT Jodhpur Data

### Run full pipeline (scrape → preprocess → train → analyze → visualize)

```bash
cd problem1/scripts
python3 run_all.py
```

### Run individual steps

```bash
# Step 1: Scrape IIT Jodhpur website
python3 problem1/scripts/collect_data.py
python3 problem1/scripts/collect_more_data.py

# Step 2: Preprocess text + generate word cloud
python3 problem1/scripts/preprocess.py

# Step 3: Train Word2Vec from scratch (CBOW + Skip-gram, all hyperparameter experiments)
python3 problem1/scripts/word2vec_scratch.py

# Step 4: Train Gensim Word2Vec for comparison
python3 problem1/scripts/word2vec_gensim.py

# Step 5: Semantic analysis + PCA/t-SNE visualizations
cd problem1/scripts && python3 analyze_and_visualize.py
```

### Skip scraping (use existing data)

```bash
cd problem1/scripts
python3 run_all.py --skip-scrape
```

### Outputs

| Directory | Contents |
|---|---|
| `problem1/data/raw/` | Scraped text documents |
| `problem1/data/processed/` | Clean corpus, sentences file, statistics |
| `problem1/models/` | Trained embeddings (18 scratch + 2 gensim) |
| `problem1/visualizations/` | Word cloud, PCA, t-SNE, loss curves |

## Problem 2: Character-Level Name Generation

### Run full pipeline (generate names → train 3 models → evaluate)

```bash
python3 problem2/scripts/train_and_evaluate.py
```

### Generate training dataset only

```bash
python3 problem2/scripts/generate_names.py
```

### Outputs

| Directory | Contents |
|---|---|
| `problem2/data/TrainingNames.txt` | 1000 Indian names |
| `problem2/models/` | Trained model checkpoints (.pt) |
| `problem2/results/` | Generated names + evaluation metrics |

## Report

The LaTeX report is at `report/report.tex`. Compile with:

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # run twice for TOC
```

Or upload to [Overleaf](https://www.overleaf.com).

## Project Structure

```
NLU_A2/
├── problem1/
│   ├── scripts/          # Data collection, preprocessing, Word2Vec, analysis
│   ├── data/             # Raw and processed corpus
│   ├── models/           # Trained embeddings
│   └── visualizations/   # All plots
├── problem2/
│   ├── scripts/          # Dataset gen, models (RNN, BLSTM, Attention), training
│   ├── data/             # TrainingNames.txt
│   ├── models/           # Saved model weights
│   └── results/          # Generated names + evaluation JSON
├── report/
│   └── report.tex
├── requirements.txt
└── README.md
```
