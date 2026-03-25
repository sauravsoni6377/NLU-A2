"""
Training and Evaluation Pipeline for Character-Level Name Generation
======================================================================
This script handles:
1. Training all three models (Vanilla RNN, BLSTM, RNN+Attention)
2. Generating names from each trained model
3. Computing quantitative metrics (novelty rate, diversity)
4. Performing qualitative analysis

All results are saved to the results/ directory for inclusion in the report.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dataset import CharVocabulary, NamesDataset, collate_fn, load_names
from models import VanillaRNN, BLSTM, AttentionRNN

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters (shared across models for fair comparison)
EMBED_DIM = 32          # Character embedding dimension
HIDDEN_SIZE = 128       # RNN hidden state size
NUM_LAYERS = 2          # Number of stacked RNN/LSTM layers
DROPOUT = 0.2           # Dropout rate for regularization
LEARNING_RATE = 0.003   # Initial learning rate (Adam optimizer)
EPOCHS = 100            # Training epochs
BATCH_SIZE = 64         # Mini-batch size
NUM_GENERATE = 200      # Number of names to generate per model for evaluation


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, train_loader, vocab, model_name, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Trains a character-level generation model using teacher forcing.

    Teacher forcing: at each timestep, the model receives the ground truth
    previous character as input (rather than its own prediction). This
    stabilizes training but can cause exposure bias.

    Uses:
    - CrossEntropyLoss: standard loss for classification at each timestep
    - Adam optimizer: adaptive learning rate for faster convergence
    - Learning rate scheduler: reduces LR when loss plateaus

    Args:
        model: PyTorch model (VanillaRNN, BLSTM, or AttentionRNN)
        train_loader: DataLoader providing batched training data
        vocab: CharVocabulary object
        model_name: String identifier for saving/logging
        epochs: Number of training passes over the data
        lr: Learning rate for Adam optimizer

    Returns:
        List of per-epoch average losses
    """
    # CrossEntropyLoss ignores PAD positions (index 0)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reduce LR by 50% if loss doesn't improve for 10 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    losses = []

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {BATCH_SIZE}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for inputs, targets, lengths in train_loader:
            optimizer.zero_grad()

            # Forward pass: get logits for each position
            logits = model(inputs, lengths)

            # Reshape for CrossEntropyLoss:
            # logits: [batch, seq_len, vocab] -> [batch*seq_len, vocab]
            # targets: [batch, seq_len] -> [batch*seq_len]
            loss = criterion(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1)
            )

            # Backward pass and parameter update
            loss.backward()

            # Gradient clipping prevents exploding gradients in RNNs
            # RNNs are prone to this due to repeated matrix multiplication
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        # Print progress every 10 epochs + generate a sample to monitor quality
        if (epoch + 1) % 10 == 0 or epoch == 0:
            sample = model.generate(vocab, temperature=0.8)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Sample: {sample}")

    return losses


# ============================================================
# NAME GENERATION
# ============================================================
def generate_names(model, vocab, num_names=NUM_GENERATE, temperature=0.8):
    """
    Generates a batch of names from a trained model.

    Uses temperature-scaled sampling to control the randomness:
    - temperature < 1: sharper distribution, more conservative names
    - temperature > 1: flatter distribution, more creative/unusual names

    Args:
        model: Trained model
        vocab: CharVocabulary
        num_names: How many names to generate
        temperature: Sampling temperature

    Returns:
        List of generated name strings
    """
    model.eval()
    names = []
    for _ in range(num_names):
        name = model.generate(vocab, max_len=20, temperature=temperature)
        # Filter out empty or single-character names
        if len(name) >= 2:
            names.append(name)
    return names


# ============================================================
# QUANTITATIVE EVALUATION (TASK 2)
# ============================================================
def evaluate_quantitative(generated_names, training_names, model_name):
    """
    Computes quantitative metrics for generated names.

    Metrics:
    1. Novelty Rate: Percentage of generated names that do NOT appear
       in the training set. Higher = model is not just memorizing.
       Novelty = |generated - training| / |generated| * 100

    2. Diversity: Ratio of unique generated names to total generated names.
       Higher = model produces varied outputs, not repeating itself.
       Diversity = |unique(generated)| / |generated| * 100

    Args:
        generated_names: List of generated name strings
        training_names: List of original training names
        model_name: String identifier for reporting

    Returns:
        Dictionary with computed metrics
    """
    # Normalize names for comparison (lowercase)
    gen_lower = [n.lower() for n in generated_names]
    train_lower = set(n.lower() for n in training_names)

    # Novelty: names not seen in training data
    novel_names = [n for n in gen_lower if n not in train_lower]
    novelty_rate = len(novel_names) / max(len(gen_lower), 1) * 100

    # Diversity: unique names among all generated
    unique_names = set(gen_lower)
    diversity = len(unique_names) / max(len(gen_lower), 1) * 100

    metrics = {
        "model": model_name,
        "total_generated": len(generated_names),
        "novel_count": len(novel_names),
        "novelty_rate": round(novelty_rate, 2),
        "unique_count": len(unique_names),
        "diversity": round(diversity, 2),
    }

    print(f"\n--- {model_name} Metrics ---")
    print(f"  Total generated:  {metrics['total_generated']}")
    print(f"  Novelty rate:     {metrics['novelty_rate']:.1f}%")
    print(f"  Diversity:        {metrics['diversity']:.1f}%")

    return metrics


# ============================================================
# QUALITATIVE ANALYSIS (TASK 3)
# ============================================================
def analyze_qualitative(generated_names, training_names, model_name):
    """
    Performs qualitative analysis of generated names.

    Analyzes:
    1. Name length distribution vs training data
    2. Common failure modes:
       - Too short names (< 3 chars)
       - Too long names (> 15 chars)
       - Unpronounceable consonant clusters
       - Repeated characters
    3. Character frequency distribution
    4. Representative samples (good and bad)

    Args:
        generated_names: List of generated names
        training_names: List of training names
        model_name: String identifier

    Returns:
        Dictionary with qualitative analysis results
    """
    # Length analysis
    gen_lengths = [len(n) for n in generated_names]
    train_lengths = [len(n) for n in training_names]

    # Identify failure modes
    too_short = [n for n in generated_names if len(n) < 3]
    too_long = [n for n in generated_names if len(n) > 15]

    # Check for repeated characters (e.g., "aaa", "kkk")
    repeated = [n for n in generated_names if any(
        n[i] == n[i+1] == n[i+2] if i+2 < len(n) else False
        for i in range(len(n))
    )]

    # Consonant cluster detection (3+ consecutive consonants without vowel)
    vowels = set('aeiouAEIOU')
    def has_bad_cluster(name):
        consonant_run = 0
        for c in name:
            if c.isalpha() and c not in vowels:
                consonant_run += 1
                if consonant_run >= 4:
                    return True
            else:
                consonant_run = 0
        return False

    unpronounceable = [n for n in generated_names if has_bad_cluster(n)]

    # Select representative good and bad samples
    good_samples = [n for n in generated_names if 3 <= len(n) <= 12 and not has_bad_cluster(n)]
    bad_samples = too_short + too_long + unpronounceable + repeated

    analysis = {
        "model": model_name,
        "avg_length_generated": round(np.mean(gen_lengths), 1) if gen_lengths else 0,
        "avg_length_training": round(np.mean(train_lengths), 1),
        "failure_modes": {
            "too_short": len(too_short),
            "too_long": len(too_long),
            "repeated_chars": len(repeated),
            "unpronounceable": len(unpronounceable),
        },
        "good_samples": good_samples[:20],
        "bad_samples": list(set(bad_samples))[:10],
        "all_generated": generated_names[:50],
    }

    print(f"\n--- {model_name} Qualitative Analysis ---")
    print(f"  Avg name length: {analysis['avg_length_generated']:.1f} "
          f"(training: {analysis['avg_length_training']:.1f})")
    print(f"  Failure modes:")
    for mode, count in analysis['failure_modes'].items():
        print(f"    {mode:20s}: {count}")
    print(f"  Good samples: {good_samples[:10]}")
    if bad_samples:
        print(f"  Bad samples:  {list(set(bad_samples))[:5]}")

    return analysis


# ============================================================
# MAIN TRAINING AND EVALUATION PIPELINE
# ============================================================
def main():
    """
    Full pipeline: load data, train all models, generate names, evaluate.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load training data ----
    names_path = os.path.join(DATA_DIR, "TrainingNames.txt")
    training_names = load_names(names_path)
    print(f"[INFO] Loaded {len(training_names)} training names")

    # ---- Build character vocabulary ----
    vocab = CharVocabulary()
    vocab.build(training_names)
    print(f"[INFO] Vocabulary size: {vocab.vocab_size} characters")
    print(f"[INFO] Characters: {sorted(vocab.char2idx.keys())}")

    # ---- Create dataset and dataloader ----
    dataset = NamesDataset(training_names, vocab)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- Define all three models ----
    models_config = {
        "Vanilla_RNN": VanillaRNN(
            vocab_size=vocab.vocab_size,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ),
        "BLSTM": BLSTM(
            vocab_size=vocab.vocab_size,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ),
        "RNN_Attention": AttentionRNN(
            vocab_size=vocab.vocab_size,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ),
    }

    # Report model architectures and parameter counts
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURES")
    print("=" * 60)
    param_info = {}
    for name, model in models_config.items():
        n_params = model.count_parameters()
        param_info[name] = n_params
        print(f"\n{name}: {n_params:,} trainable parameters")
        print(f"  Architecture: {model}")

    all_results = {
        "hyperparameters": {
            "embed_dim": EMBED_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        "parameter_counts": param_info,
        "models": {},
    }

    # ---- Train, Generate, and Evaluate each model ----
    for model_name, model in models_config.items():
        # Train the model
        losses = train_model(model, train_loader, vocab, model_name)

        # Save trained model weights
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)

        # Generate names at different temperatures
        print(f"\n[GENERATING] {model_name}...")
        generated_default = generate_names(model, vocab, NUM_GENERATE, temperature=0.8)
        generated_low_temp = generate_names(model, vocab, 50, temperature=0.5)
        generated_high_temp = generate_names(model, vocab, 50, temperature=1.2)

        # Save generated names to file
        gen_path = os.path.join(RESULTS_DIR, f"{model_name}_generated.txt")
        with open(gen_path, "w") as f:
            for name in generated_default:
                f.write(name + "\n")

        # Quantitative evaluation
        quant_metrics = evaluate_quantitative(generated_default, training_names, model_name)

        # Qualitative analysis
        qual_analysis = analyze_qualitative(generated_default, training_names, model_name)

        # Store results
        all_results["models"][model_name] = {
            "losses": losses,
            "quantitative": quant_metrics,
            "qualitative": qual_analysis,
            "samples_temp_0.5": generated_low_temp[:20],
            "samples_temp_0.8": generated_default[:20],
            "samples_temp_1.2": generated_high_temp[:20],
        }

    # ---- Save comprehensive results ----
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[INFO] All results saved to {results_path}")

    # ---- Print comparative summary ----
    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Params':>10} {'Final Loss':>12} {'Novelty':>10} {'Diversity':>10}")
    print("-" * 62)
    for model_name in models_config:
        r = all_results["models"][model_name]
        final_loss = r["losses"][-1] if r["losses"] else float('nan')
        novelty = r["quantitative"]["novelty_rate"]
        diversity = r["quantitative"]["diversity"]
        params = param_info[model_name]
        print(f"{model_name:<20} {params:>10,} {final_loss:>12.4f} {novelty:>9.1f}% {diversity:>9.1f}%")

    print("\n[DONE] Training and evaluation complete!")


if __name__ == "__main__":
    main()
