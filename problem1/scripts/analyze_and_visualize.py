"""
Semantic Analysis and Visualization Script
=============================================
This script performs:
1. Task 3: Semantic analysis using the trained Word2Vec models
   - Top 5 nearest neighbors for specified words
   - Word analogy experiments
2. Task 4: Visualization using PCA and t-SNE
   - 2D projections of word embeddings
   - Cluster visualizations for CBOW vs Skip-gram comparison

All results are saved as figures in the visualizations/ directory
and as JSON for inclusion in the LaTeX report.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import our scratch implementation utilities
from word2vec_scratch import load_model, find_nearest_neighbors, analogy, cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)


# ============================================================
# TASK 3: SEMANTIC ANALYSIS
# ============================================================
def run_semantic_analysis(embeddings, vocab, model_name):
    """
    Performs semantic analysis on trained word embeddings.

    Part 1: Nearest Neighbors
    - Finds top 5 most similar words for: research, student, phd, exam
    - Uses cosine similarity as the distance metric

    Part 2: Word Analogies
    - Tests vector arithmetic: A:B :: C:?
    - At least 3 analogy experiments as required

    Args:
        embeddings: numpy array [vocab_size, embed_dim]
        vocab: Vocabulary object with word2idx/idx2word
        model_name: Name string for labeling results

    Returns:
        Dictionary containing all analysis results
    """
    results = {"nearest_neighbors": {}, "analogies": {}}

    # ---- Part 1: Nearest Neighbors ----
    # Words specified in the assignment (exam appears twice in assignment, likely a typo)
    # "exam" may not be in vocabulary due to min_count filtering, so we include
    # "semester" and "faculty" as additional relevant alternatives
    query_words = ["research", "student", "phd", "semester", "faculty"]

    print(f"\n{'='*50}")
    print(f"Semantic Analysis: {model_name}")
    print(f"{'='*50}")

    print("\n--- Top 5 Nearest Neighbors (Cosine Similarity) ---")
    for word in query_words:
        neighbors = find_nearest_neighbors(word, embeddings, vocab, top_k=5)
        results["nearest_neighbors"][word] = [
            {"word": w, "similarity": float(s)} for w, s in neighbors
        ]
        if neighbors:
            print(f"\n  '{word}':")
            for w, s in neighbors:
                print(f"    {w:20s} : {s:.4f}")

    # ---- Part 2: Analogy Experiments ----
    # At least 3 analogies as required by the assignment
    analogy_tests = [
        # Analogy 1: UG:BTech :: PG:? (expected: MTech) - as specified in assignment
        ("ug", "btech", "pg", "Degree level analogy"),
        # Analogy 2: department:head :: institute:? (expected: director)
        ("department", "head", "institute", "Leadership hierarchy analogy"),
        # Analogy 3: student:campus :: faculty:? (expected: department/office)
        ("student", "campus", "faculty", "Role-place analogy"),
        # Analogy 4: btech:ug :: mtech:? (expected: pg)
        ("btech", "ug", "mtech", "Degree abbreviation analogy"),
    ]

    print("\n--- Word Analogy Experiments ---")
    for word_a, word_b, word_c, description in analogy_tests:
        print(f"\n  {description}: {word_a} : {word_b} :: {word_c} : ?")
        result = analogy(word_a, word_b, word_c, embeddings, vocab, top_k=5)
        key = f"{word_a}:{word_b}::{word_c}:?"
        results["analogies"][key] = {
            "description": description,
            "results": [{"word": w, "similarity": float(s)} for w, s in result],
        }
        if result:
            for w, s in result:
                print(f"    {w:20s} : {s:.4f}")

    return results


# ============================================================
# TASK 4: VISUALIZATION
# ============================================================
def visualize_embeddings_pca(embeddings, vocab, model_name, word_groups=None):
    """
    Projects word embeddings to 2D using PCA and creates scatter plots.

    PCA (Principal Component Analysis) finds the directions of maximum
    variance in the high-dimensional embedding space and projects onto
    the top 2 principal components.

    Args:
        embeddings: numpy array [vocab_size, embed_dim]
        vocab: Vocabulary object
        model_name: String for plot title and filename
        word_groups: Dict mapping group_name -> list of words to highlight
    """
    if word_groups is None:
        # Default word groups covering different semantic categories
        word_groups = {
            "Academic Programs": ["btech", "mtech", "phd", "msc", "mba", "ug", "pg"],
            "Departments": ["cse", "electrical", "mechanical", "mathematics", "physics",
                          "chemistry", "humanities"],
            "Research": ["research", "publication", "journal", "conference", "paper",
                        "thesis", "dissertation"],
            "Campus Life": ["student", "hostel", "campus", "library", "sports",
                          "club", "placement"],
            "Administration": ["director", "dean", "hod", "professor", "faculty",
                             "senate", "committee"],
        }

    # Collect all words that exist in vocabulary
    all_words = []
    word_to_group = {}
    for group_name, words in word_groups.items():
        for word in words:
            if word in vocab.word2idx:
                all_words.append(word)
                word_to_group[word] = group_name

    if len(all_words) < 5:
        print(f"[WARNING] Too few words found in vocabulary for visualization ({len(all_words)})")
        # Fallback: use top 100 most frequent words
        all_words = [vocab.idx2word[i] for i in range(min(100, vocab.vocab_size))]
        word_to_group = {w: "Frequent" for w in all_words}

    # Get embeddings for selected words
    word_indices = [vocab.word2idx[w] for w in all_words]
    selected_embeddings = embeddings[word_indices]

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(selected_embeddings)

    # Calculate variance explained by each component
    var_explained = pca.explained_variance_ratio_

    # Create scatter plot with color-coded groups
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define colors for each group
    colors = plt.cm.Set2(np.linspace(0, 1, len(set(word_to_group.values()))))
    group_colors = {g: colors[i] for i, g in enumerate(sorted(set(word_to_group.values())))}

    # Plot each word, colored by its group
    for i, word in enumerate(all_words):
        group = word_to_group[word]
        color = group_colors[group]
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], c=[color], s=60, alpha=0.7)
        ax.annotate(
            word,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=8,
            alpha=0.85,
            ha="center",
            va="bottom",
        )

    # Add legend for groups
    for group, color in group_colors.items():
        ax.scatter([], [], c=[color], label=group, s=60)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    ax.set_title(
        f"PCA Projection - {model_name}\n"
        f"(PC1: {var_explained[0]:.1%} variance, PC2: {var_explained[1]:.1%} variance)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(f"Principal Component 1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"Principal Component 2 ({var_explained[1]:.1%})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(VIS_DIR, f"pca_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] PCA plot saved: {save_path}")


def visualize_embeddings_tsne(embeddings, vocab, model_name, word_groups=None):
    """
    Projects word embeddings to 2D using t-SNE and creates scatter plots.

    t-SNE (t-distributed Stochastic Neighbor Embedding) is a non-linear
    dimensionality reduction technique particularly good at preserving
    local structure (nearby points in high-D stay nearby in 2D).

    Unlike PCA, t-SNE emphasizes cluster structure over global geometry.

    Args:
        embeddings: numpy array [vocab_size, embed_dim]
        vocab: Vocabulary object
        model_name: String for plot title and filename
        word_groups: Dict mapping group_name -> list of words
    """
    if word_groups is None:
        word_groups = {
            "Academic Programs": ["btech", "mtech", "phd", "msc", "mba", "ug", "pg"],
            "Departments": ["cse", "electrical", "mechanical", "mathematics", "physics",
                          "chemistry", "humanities"],
            "Research": ["research", "publication", "journal", "conference", "paper",
                        "thesis", "dissertation"],
            "Campus Life": ["student", "hostel", "campus", "library", "sports",
                          "club", "placement"],
            "Administration": ["director", "dean", "hod", "professor", "faculty",
                             "senate", "committee"],
        }

    # Collect valid words
    all_words = []
    word_to_group = {}
    for group_name, words in word_groups.items():
        for word in words:
            if word in vocab.word2idx:
                all_words.append(word)
                word_to_group[word] = group_name

    if len(all_words) < 5:
        all_words = [vocab.idx2word[i] for i in range(min(100, vocab.vocab_size))]
        word_to_group = {w: "Frequent" for w in all_words}

    word_indices = [vocab.word2idx[w] for w in all_words]
    selected_embeddings = embeddings[word_indices]

    # Apply t-SNE with perplexity suited to our data size
    # Perplexity should be less than the number of points
    perplexity = min(30, len(all_words) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        learning_rate="auto",
        init="pca",  # Initialize with PCA for stability
    )
    coords_2d = tsne.fit_transform(selected_embeddings)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = plt.cm.Set2(np.linspace(0, 1, len(set(word_to_group.values()))))
    group_colors = {g: colors[i] for i, g in enumerate(sorted(set(word_to_group.values())))}

    for i, word in enumerate(all_words):
        group = word_to_group[word]
        color = group_colors[group]
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], c=[color], s=60, alpha=0.7)
        ax.annotate(
            word,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=8,
            alpha=0.85,
            ha="center",
            va="bottom",
        )

    for group, color in group_colors.items():
        ax.scatter([], [], c=[color], label=group, s=60)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    ax.set_title(f"t-SNE Projection - {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(VIS_DIR, f"tsne_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] t-SNE plot saved: {save_path}")


def plot_training_losses():
    """
    Plots training loss curves for all trained models.
    Useful for comparing convergence behavior across different
    hyperparameter settings and model types.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Collect all model metadata files
    experiment_groups = {
        "Embedding Dimension": [],
        "Window Size": [],
        "Negative Samples": [],
    }

    for model_dir_name in sorted(os.listdir(MODEL_DIR)):
        model_path = os.path.join(MODEL_DIR, model_dir_name)
        if not os.path.isdir(model_path):
            continue

        # Look for metadata files
        for fname in os.listdir(model_path):
            if fname.endswith("_metadata.json"):
                meta_path = os.path.join(model_path, fname)
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                label = model_dir_name
                losses = meta.get("losses", [])
                config = meta.get("config", {})

                if not losses:
                    continue

                # Categorize by which hyperparameter was varied
                if "dim" in model_dir_name:
                    experiment_groups["Embedding Dimension"].append((label, losses))
                elif "win" in model_dir_name:
                    experiment_groups["Window Size"].append((label, losses))
                elif "neg" in model_dir_name:
                    experiment_groups["Negative Samples"].append((label, losses))

    # Plot each experiment group
    titles = ["Embedding Dimension", "Window Size", "Negative Samples"]
    for ax, title in zip(axes, titles):
        group = experiment_groups.get(title, [])
        if group:
            for label, losses in group:
                ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3, label=label)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Loss - Varying {title}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title(f"No data for {title}")

    plt.tight_layout()
    save_path = os.path.join(VIS_DIR, "training_losses.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Training loss plot saved: {save_path}")


def compare_cbow_vs_skipgram():
    """
    Creates a side-by-side comparison visualization of CBOW vs Skip-gram.
    Uses the default configuration models (dim=100, win=5, neg=5).
    """
    # Load both models with default config
    models_to_compare = [
        ("skipgram_dim100", "Skip-gram (scratch)"),
        ("cbow_dim100", "CBOW (scratch)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for idx, (model_key, title) in enumerate(models_to_compare):
        model_path = os.path.join(MODEL_DIR, model_key)
        if not os.path.isdir(model_path):
            print(f"[WARNING] Model not found: {model_path}")
            continue

        # Determine model type from key
        model_type = "skipgram" if "skipgram" in model_key else "cbow"
        embeddings, vocab = load_model(model_type, model_path)

        # Define word groups for visualization
        word_groups = {
            "Academic Programs": ["btech", "mtech", "phd", "msc", "mba", "ug", "pg"],
            "Departments": ["cse", "electrical", "mechanical", "mathematics", "physics",
                          "chemistry", "humanities"],
            "Research": ["research", "publication", "journal", "conference", "paper",
                        "thesis", "dissertation"],
            "Campus Life": ["student", "hostel", "campus", "library", "sports",
                          "club", "placement"],
            "Administration": ["director", "dean", "hod", "professor", "faculty",
                             "senate", "committee"],
        }

        # Collect valid words
        all_words = []
        word_to_group = {}
        for group_name, words in word_groups.items():
            for word in words:
                if word in vocab.word2idx:
                    all_words.append(word)
                    word_to_group[word] = group_name

        if len(all_words) < 5:
            all_words = [vocab.idx2word[i] for i in range(min(80, vocab.vocab_size))]
            word_to_group = {w: "Frequent" for w in all_words}

        word_indices = [vocab.word2idx[w] for w in all_words]
        selected_embeddings = embeddings[word_indices]

        # Use t-SNE for comparison plot
        perplexity = min(20, len(all_words) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     n_iter=1000, learning_rate="auto", init="pca")
        coords_2d = tsne.fit_transform(selected_embeddings)

        ax = axes[idx]
        colors_map = plt.cm.Set2(np.linspace(0, 1, len(set(word_to_group.values()))))
        group_colors = {g: colors_map[i] for i, g in enumerate(sorted(set(word_to_group.values())))}

        for i, word in enumerate(all_words):
            group = word_to_group[word]
            color = group_colors[group]
            ax.scatter(coords_2d[i, 0], coords_2d[i, 1], c=[color], s=60, alpha=0.7)
            ax.annotate(word, (coords_2d[i, 0], coords_2d[i, 1]),
                       fontsize=7, alpha=0.85, ha="center", va="bottom")

        for group, color in group_colors.items():
            ax.scatter([], [], c=[color], label=group, s=60)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle("CBOW vs Skip-gram: t-SNE Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(VIS_DIR, "cbow_vs_skipgram_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIS] Comparison plot saved: {save_path}")


def main():
    """
    Main function that runs all analysis and visualization tasks.
    """
    print("=" * 60)
    print("Semantic Analysis & Visualization")
    print("=" * 60)

    all_analysis_results = {}

    # ---- Analyze default models (dim=100, win=5, neg=5) ----
    for model_key in ["skipgram_dim100", "cbow_dim100"]:
        model_path = os.path.join(MODEL_DIR, model_key)
        if not os.path.isdir(model_path):
            print(f"[SKIP] Model directory not found: {model_path}")
            continue

        model_type = "skipgram" if "skipgram" in model_key else "cbow"
        embeddings, vocab = load_model(model_type, model_path)

        # Run semantic analysis (Task 3)
        results = run_semantic_analysis(embeddings, vocab, f"{model_type.upper()} (scratch)")
        all_analysis_results[model_key] = results

        # Generate PCA and t-SNE visualizations (Task 4)
        visualize_embeddings_pca(embeddings, vocab, f"{model_type.upper()} Scratch")
        visualize_embeddings_tsne(embeddings, vocab, f"{model_type.upper()} Scratch")

    # Save all analysis results
    results_path = os.path.join(VIS_DIR, "semantic_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(all_analysis_results, f, indent=2)
    print(f"\n[INFO] Analysis results saved to {results_path}")

    # Plot training loss curves across experiments
    plot_training_losses()

    # Create CBOW vs Skip-gram comparison plot
    compare_cbow_vs_skipgram()

    print("\n[DONE] All analysis and visualization complete!")


if __name__ == "__main__":
    main()
