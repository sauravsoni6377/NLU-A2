"""
Word2Vec Training using Gensim (Library-based Comparison)
==========================================================
This script trains Word2Vec models using the Gensim library for comparison
with our from-scratch PyTorch implementation.

We train both CBOW and Skip-gram models with the same hyperparameters
used in the scratch implementation to enable a fair comparison.

The professor requires: "For Word2Vec do from scratch and compare with existing ones."
This script provides the "existing ones" part of that comparison.
"""

import os
import json
import numpy as np
from gensim.models import Word2Vec

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "gensim")


def load_sentences():
    """
    Loads preprocessed sentences from the corpus file.
    Each line is a space-separated sequence of tokens.

    Returns:
        List of lists of tokens
    """
    sentences_path = os.path.join(PROCESSED_DATA_DIR, "sentences.txt")
    sentences = []
    with open(sentences_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                sentences.append(tokens)
    print(f"[INFO] Loaded {len(sentences)} sentences for Gensim training")
    return sentences


def train_gensim_models(sentences):
    """
    Trains CBOW and Skip-gram Word2Vec models using Gensim.

    Uses the same default hyperparameters as the scratch implementation:
    - embedding_dim=100, window=5, negative=5, min_count=3, epochs=15

    Gensim's Word2Vec parameter mapping:
    - sg=0 -> CBOW, sg=1 -> Skip-gram
    - vector_size -> embedding dimension
    - window -> context window size
    - negative -> number of negative samples
    - min_count -> minimum word frequency
    - epochs -> number of training iterations

    Args:
        sentences: List of tokenized sentences

    Returns:
        Dictionary mapping model name to trained Gensim model
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    models = {}

    # Train CBOW model (sg=0 means CBOW in Gensim)
    print("\n[GENSIM] Training CBOW model...")
    cbow_model = Word2Vec(
        sentences=sentences,
        vector_size=100,       # Same embedding dimension as scratch
        window=5,              # Same context window
        min_count=3,           # Same minimum word count
        sg=0,                  # 0 = CBOW
        negative=5,            # Same number of negative samples
        epochs=15,             # Same number of epochs
        seed=42,               # For reproducibility
        workers=4,             # Use multiple cores for speed
    )
    cbow_model.save(os.path.join(MODEL_DIR, "cbow_gensim.model"))
    models["gensim_cbow"] = cbow_model
    print(f"  Vocabulary size: {len(cbow_model.wv)}")

    # Train Skip-gram model (sg=1 means Skip-gram in Gensim)
    print("\n[GENSIM] Training Skip-gram model...")
    sg_model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=3,
        sg=1,                  # 1 = Skip-gram
        negative=5,
        epochs=15,
        seed=42,
        workers=4,
    )
    sg_model.save(os.path.join(MODEL_DIR, "skipgram_gensim.model"))
    models["gensim_skipgram"] = sg_model
    print(f"  Vocabulary size: {len(sg_model.wv)}")

    return models


def evaluate_gensim_models(models):
    """
    Evaluates Gensim models using the same queries as the scratch models.
    Reports nearest neighbors and analogy results for comparison.

    Args:
        models: Dictionary mapping model name to Gensim Word2Vec model
    """
    # Words to query for nearest neighbors (same as Task 3)
    query_words = ["research", "student", "phd", "exam"]

    # Analogy tests (same as Task 3)
    analogies = [
        ("ug", "btech", "pg"),             # UG:BTech :: PG:?
        ("department", "head", "institute"),  # department:head :: institute:?
        ("btech", "ug", "mtech"),          # BTech:UG :: MTech:?
    ]

    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluation: {model_name}")
        print(f"{'='*50}")

        model_results = {"nearest_neighbors": {}, "analogies": {}}

        # Nearest neighbors
        print("\n--- Nearest Neighbors ---")
        for word in query_words:
            if word in model.wv:
                neighbors = model.wv.most_similar(word, topn=5)
                model_results["nearest_neighbors"][word] = neighbors
                print(f"\n  '{word}':")
                for neighbor, score in neighbors:
                    print(f"    {neighbor:20s} : {score:.4f}")
            else:
                print(f"\n  '{word}' not in vocabulary")

        # Analogies
        print("\n--- Analogies ---")
        for a, b, c in analogies:
            if all(w in model.wv for w in [a, b, c]):
                try:
                    result = model.wv.most_similar(
                        positive=[b, c], negative=[a], topn=5
                    )
                    key = f"{a}:{b}::{c}:?"
                    model_results["analogies"][key] = result
                    print(f"\n  {a} : {b} :: {c} : ?")
                    for word, score in result:
                        print(f"    {word:20s} : {score:.4f}")
                except KeyError as e:
                    print(f"  Analogy failed: {e}")
            else:
                missing = [w for w in [a, b, c] if w not in model.wv]
                print(f"  Skipping analogy: missing words {missing}")

        results[model_name] = model_results

    # Save results
    # Convert numpy floats for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        if isinstance(obj, tuple):
            return [convert(i) for i in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    results_path = os.path.join(MODEL_DIR, "gensim_results.json")
    with open(results_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n[INFO] Gensim results saved to {results_path}")


def main():
    """
    Main function: loads data, trains Gensim models, and evaluates them.
    """
    print("=" * 60)
    print("Gensim Word2Vec Training & Evaluation")
    print("=" * 60)

    sentences = load_sentences()
    if not sentences:
        print("[ERROR] No sentences found. Run preprocess.py first.")
        return

    models = train_gensim_models(sentences)
    evaluate_gensim_models(models)

    print("\n[DONE] Gensim training and evaluation complete!")


if __name__ == "__main__":
    main()
