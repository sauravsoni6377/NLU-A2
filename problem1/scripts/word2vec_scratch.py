"""
Word2Vec Implementation from Scratch using PyTorch
=====================================================
This module implements both CBOW and Skip-gram with Negative Sampling
from scratch, using PyTorch for automatic differentiation and gradient descent.

Architecture:
- CBOW: Predicts the center word given context words
- Skip-gram: Predicts context words given the center word
- Both use Negative Sampling (instead of full softmax) for efficiency

Key components:
1. Vocabulary builder with word-to-index mapping
2. Training data generator (context-target pairs with negative samples)
3. CBOW model class (PyTorch nn.Module)
4. Skip-gram model class (PyTorch nn.Module)
5. Training loop with configurable hyperparameters
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Set random seeds for reproducibility across all libraries
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# VOCABULARY BUILDER
# ============================================================
class Vocabulary:
    """
    Builds and manages the vocabulary for Word2Vec training.

    Handles:
    - Building word-to-index and index-to-word mappings
    - Computing word frequencies for negative sampling distribution
    - Subsampling frequent words (as in original Word2Vec paper)
    - Computing unigram distribution raised to 3/4 power for negative sampling
    """

    def __init__(self, min_count=3):
        """
        Args:
            min_count: Minimum frequency for a word to be included in vocabulary.
                       Words appearing fewer than min_count times are discarded
                       to reduce noise and vocabulary size.
        """
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0

        # Negative sampling distribution: unigram^(3/4) as in Mikolov et al.
        self.neg_sample_probs = None

    def build(self, sentences):
        """
        Builds vocabulary from a list of tokenized sentences.

        Steps:
        1. Count all word occurrences across the corpus
        2. Filter out words below min_count threshold
        3. Create word <-> index mappings
        4. Compute negative sampling probability distribution

        Args:
            sentences: List of lists of tokens (each inner list is a sentence)
        """
        # Count word frequencies across all sentences
        for sentence in sentences:
            self.word_counts.update(sentence)

        # Filter words by minimum count and create mappings
        # This removes rare/noisy words from the vocabulary
        idx = 0
        for word, count in self.word_counts.most_common():
            if count >= self.min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        self.vocab_size = len(self.word2idx)
        print(f"[VOCAB] Total unique words: {len(self.word_counts)}")
        print(f"[VOCAB] Vocabulary size (min_count={self.min_count}): {self.vocab_size}")

        # Compute negative sampling distribution
        # Following Mikolov et al., we raise unigram frequencies to the 3/4 power
        # This smooths the distribution, giving rare words slightly higher probability
        # of being selected as negative samples
        self._compute_neg_sampling_dist()

    def _compute_neg_sampling_dist(self):
        """
        Computes the negative sampling probability distribution.

        The probability of sampling word w as a negative sample is:
            P(w) = f(w)^(3/4) / sum(f(w_i)^(3/4))

        where f(w) is the frequency of word w. The 3/4 exponent is used
        (as in the original paper) to flatten the distribution slightly,
        giving rare words a better chance of being sampled as negatives.
        """
        counts = np.zeros(self.vocab_size)
        for word, idx in self.word2idx.items():
            counts[idx] = self.word_counts[word]

        # Raise to 3/4 power to smooth the distribution
        powered = np.power(counts, 0.75)
        self.neg_sample_probs = powered / powered.sum()

    def get_negative_samples(self, num_samples, exclude_idx=None):
        """
        Draws negative samples from the noise distribution.

        Args:
            num_samples: Number of negative samples to draw
            exclude_idx: Index to exclude (the true target word)

        Returns:
            numpy array of sampled word indices
        """
        samples = np.random.choice(
            self.vocab_size,
            size=num_samples,
            p=self.neg_sample_probs
        )

        # Resample any that accidentally match the target
        if exclude_idx is not None:
            for i in range(len(samples)):
                while samples[i] == exclude_idx:
                    samples[i] = np.random.choice(self.vocab_size, p=self.neg_sample_probs)

        return samples

    def encode_sentence(self, sentence):
        """
        Converts a sentence (list of tokens) to a list of word indices.
        Words not in vocabulary are silently skipped.

        Args:
            sentence: List of string tokens

        Returns:
            List of integer indices
        """
        return [self.word2idx[w] for w in sentence if w in self.word2idx]


# ============================================================
# TRAINING DATA GENERATOR
# ============================================================
def generate_training_pairs(encoded_sentences, window_size, mode="skipgram"):
    """
    Generates (context, target) training pairs from encoded sentences.

    For Skip-gram: Each pair is (center_word, context_word)
    For CBOW: Each pair is ([context_words], center_word)

    The window size determines how many words on each side of the center
    word are considered as context. For each center word position, we
    randomly sample the actual window size from [1, window_size] as in
    the original Word2Vec implementation.

    Args:
        encoded_sentences: List of lists of word indices
        window_size: Maximum context window size on each side
        mode: "skipgram" or "cbow"

    Returns:
        List of (input, target) pairs
    """
    pairs = []

    for sentence in encoded_sentences:
        sent_len = len(sentence)
        if sent_len < 2:
            continue

        for i in range(sent_len):
            # Randomly reduce window size for each position (as in original paper)
            # This gives closer words higher effective weight
            actual_window = random.randint(1, window_size)

            # Get context word indices within the window
            context_indices = []
            for j in range(max(0, i - actual_window), min(sent_len, i + actual_window + 1)):
                if j != i:  # Exclude the center word itself
                    context_indices.append(sentence[j])

            if not context_indices:
                continue

            if mode == "skipgram":
                # Skip-gram: each (center, context_word) is a separate pair
                for ctx_word in context_indices:
                    pairs.append((sentence[i], ctx_word))
            else:
                # CBOW: (list_of_context_words, center_word) as one pair
                pairs.append((context_indices, sentence[i]))

    return pairs


# ============================================================
# SKIP-GRAM MODEL WITH NEGATIVE SAMPLING
# ============================================================
class SkipGramNS(nn.Module):
    """
    Skip-gram model with Negative Sampling (SGNS).

    Architecture:
    - Two embedding matrices: input (center word) and output (context word)
    - For a given (center, context) pair, the model tries to maximize the
      dot product between the center embedding and the true context embedding,
      while minimizing it for randomly sampled negative words

    Loss function (Negative Sampling):
        L = -log(sigma(v_c . v_w)) - sum_k[log(sigma(-v_c . v_nk))]
    where:
        v_c = center word embedding
        v_w = true context word embedding
        v_nk = k-th negative sample embedding
        sigma = sigmoid function
    """

    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimensionality of word embeddings
        """
        super(SkipGramNS, self).__init__()

        # Input embeddings (for center/target words)
        # Initialized with uniform distribution in [-0.5/dim, 0.5/dim]
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.center_embeddings.weight.data.uniform_(
            -0.5 / embedding_dim, 0.5 / embedding_dim
        )

        # Output embeddings (for context words and negative samples)
        # Initialized to zeros (as in original Word2Vec implementation)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings.weight.data.zero_()

    def forward(self, center_idx, pos_context_idx, neg_context_idx):
        """
        Forward pass computing the negative sampling loss.

        Args:
            center_idx: Tensor of center word indices [batch_size]
            pos_context_idx: Tensor of positive context word indices [batch_size]
            neg_context_idx: Tensor of negative sample indices [batch_size, num_neg]

        Returns:
            Scalar loss value (negative sampling objective)
        """
        # Get embeddings for center words: [batch_size, embed_dim]
        center_emb = self.center_embeddings(center_idx)

        # Get embeddings for positive context words: [batch_size, embed_dim]
        pos_emb = self.context_embeddings(pos_context_idx)

        # Get embeddings for negative samples: [batch_size, num_neg, embed_dim]
        neg_emb = self.context_embeddings(neg_context_idx)

        # Positive score: dot product between center and true context
        # Result shape: [batch_size]
        pos_score = torch.sum(center_emb * pos_emb, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        # Negative score: dot product between center and each negative sample
        # center_emb.unsqueeze(1): [batch_size, 1, embed_dim]
        # neg_emb: [batch_size, num_neg, embed_dim]
        # bmm result: [batch_size, 1, num_neg] -> squeeze to [batch_size, num_neg]
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        # Total loss: negative of (positive_loss + negative_loss)
        # We negate because we want to maximize the objective
        loss = -(pos_loss + neg_loss).mean()

        return loss

    def get_embeddings(self):
        """
        Returns the learned word embeddings (center embeddings).

        Returns:
            numpy array of shape [vocab_size, embedding_dim]
        """
        return self.center_embeddings.weight.data.cpu().numpy()


# ============================================================
# CBOW MODEL WITH NEGATIVE SAMPLING
# ============================================================
class CBOWNS(nn.Module):
    """
    Continuous Bag of Words (CBOW) model with Negative Sampling.

    Architecture:
    - Input: Multiple context word embeddings averaged together
    - Output: Prediction of the center word
    - Uses negative sampling instead of full softmax

    Unlike Skip-gram which predicts context from center, CBOW predicts
    the center word from the average of its context word embeddings.

    Loss function:
        L = -log(sigma(v_avg_ctx . v_center)) - sum_k[log(sigma(-v_avg_ctx . v_nk))]
    """

    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimensionality of word embeddings
        """
        super(CBOWNS, self).__init__()

        # Context word embeddings (input side)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings.weight.data.uniform_(
            -0.5 / embedding_dim, 0.5 / embedding_dim
        )

        # Center/target word embeddings (output side)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.center_embeddings.weight.data.zero_()

    def forward(self, context_indices, pos_center_idx, neg_indices):
        """
        Forward pass for CBOW with negative sampling.

        Args:
            context_indices: Tensor of context word indices [batch_size, max_ctx_len]
                             Padded with -1 for variable-length contexts
            pos_center_idx: Tensor of true center word indices [batch_size]
            neg_indices: Tensor of negative sample indices [batch_size, num_neg]

        Returns:
            Scalar loss value
        """
        # Create mask for valid context positions (non-padding)
        mask = (context_indices >= 0).float()  # [batch_size, max_ctx_len]

        # Clamp indices to valid range (padding indices become 0, but will be masked)
        safe_indices = context_indices.clamp(min=0)

        # Get context embeddings and compute their mean (masked average)
        # [batch_size, max_ctx_len, embed_dim]
        ctx_emb = self.context_embeddings(safe_indices)

        # Apply mask to zero out padding positions before averaging
        # mask.unsqueeze(2): [batch_size, max_ctx_len, 1]
        ctx_emb = ctx_emb * mask.unsqueeze(2)

        # Average over context positions (divide by actual number of context words)
        ctx_mean = ctx_emb.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Get positive (true center) embeddings: [batch_size, embed_dim]
        pos_emb = self.center_embeddings(pos_center_idx)

        # Get negative sample embeddings: [batch_size, num_neg, embed_dim]
        neg_emb = self.center_embeddings(neg_indices)

        # Positive score: dot product of averaged context with true center
        pos_score = torch.sum(ctx_mean * pos_emb, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        # Negative score: dot product of averaged context with negative samples
        neg_score = torch.bmm(neg_emb, ctx_mean.unsqueeze(2)).squeeze(2)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def get_embeddings(self):
        """
        Returns the learned word embeddings (context embeddings for CBOW).

        Returns:
            numpy array of shape [vocab_size, embedding_dim]
        """
        return self.context_embeddings.weight.data.cpu().numpy()


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_word2vec(
    sentences,
    model_type="skipgram",
    embedding_dim=100,
    window_size=5,
    num_negative=5,
    min_count=3,
    learning_rate=0.025,
    epochs=10,
    batch_size=512,
    device="cpu",
):
    """
    Trains a Word2Vec model (either CBOW or Skip-gram) from scratch.

    This function orchestrates the full training pipeline:
    1. Build vocabulary from sentences
    2. Generate training pairs (context-target)
    3. Initialize the model (CBOW or Skip-gram)
    4. Train with SGD using negative sampling loss
    5. Return trained embeddings and vocabulary

    Args:
        sentences: List of tokenized sentences (list of lists of strings)
        model_type: "skipgram" or "cbow"
        embedding_dim: Size of word embedding vectors
        window_size: Context window size (words on each side)
        num_negative: Number of negative samples per positive pair
        min_count: Minimum word frequency to include in vocabulary
        learning_rate: Initial learning rate for SGD
        epochs: Number of training passes over the data
        batch_size: Mini-batch size for gradient updates
        device: "cpu" or "cuda"

    Returns:
        Tuple of (embeddings_matrix, vocabulary_object, training_losses)
    """
    print(f"\n{'='*60}")
    print(f"Training Word2Vec ({model_type.upper()})")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Window size:   {window_size}")
    print(f"  Neg samples:   {num_negative}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"{'='*60}")

    # Step 1: Build vocabulary
    vocab = Vocabulary(min_count=min_count)
    vocab.build(sentences)

    if vocab.vocab_size < 10:
        print("[ERROR] Vocabulary too small! Need more training data.")
        return None, None, []

    # Step 2: Encode sentences (convert words to indices)
    encoded_sentences = [vocab.encode_sentence(s) for s in sentences]
    # Remove empty sentences
    encoded_sentences = [s for s in encoded_sentences if len(s) >= 2]
    print(f"[INFO] Encoded {len(encoded_sentences)} sentences")

    # Step 3: Initialize model
    if model_type == "skipgram":
        model = SkipGramNS(vocab.vocab_size, embedding_dim).to(device)
    else:
        model = CBOWNS(vocab.vocab_size, embedding_dim).to(device)

    # Use SGD optimizer (as in original Word2Vec; Adam also works well)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 4: Training loop
    losses = []

    for epoch in range(epochs):
        # Regenerate training pairs each epoch (randomized window sizes)
        pairs = generate_training_pairs(encoded_sentences, window_size, mode=model_type)
        random.shuffle(pairs)

        epoch_loss = 0.0
        num_batches = 0

        # Process in mini-batches
        for batch_start in tqdm(
            range(0, len(pairs), batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
        ):
            batch = pairs[batch_start : batch_start + batch_size]

            if model_type == "skipgram":
                # Prepare skip-gram batch tensors
                center_batch = torch.tensor(
                    [p[0] for p in batch], dtype=torch.long, device=device
                )
                context_batch = torch.tensor(
                    [p[1] for p in batch], dtype=torch.long, device=device
                )

                # Generate negative samples for each pair in the batch
                neg_batch = torch.tensor(
                    np.array([
                        vocab.get_negative_samples(num_negative, exclude_idx=p[1])
                        for p in batch
                    ]),
                    dtype=torch.long,
                    device=device,
                )

                # Forward pass: compute negative sampling loss
                loss = model(center_batch, context_batch, neg_batch)

            else:
                # Prepare CBOW batch tensors
                # Context lists have variable lengths, so we pad to max length
                context_lists = [p[0] for p in batch]
                max_ctx = max(len(c) for c in context_lists)

                # Pad shorter contexts with -1 (will be masked in forward pass)
                padded_contexts = [
                    c + [-1] * (max_ctx - len(c)) for c in context_lists
                ]

                context_batch = torch.tensor(
                    padded_contexts, dtype=torch.long, device=device
                )
                center_batch = torch.tensor(
                    [p[1] for p in batch], dtype=torch.long, device=device
                )

                # Generate negative samples
                neg_batch = torch.tensor(
                    np.array([
                        vocab.get_negative_samples(num_negative, exclude_idx=p[1])
                        for p in batch
                    ]),
                    dtype=torch.long,
                    device=device,
                )

                # Forward pass
                loss = model(context_batch, center_batch, neg_batch)

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Extract final embeddings
    embeddings = model.get_embeddings()
    print(f"\n[DONE] Training complete. Embedding shape: {embeddings.shape}")

    return embeddings, vocab, losses


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Cosine similarity = (A . B) / (||A|| * ||B||)
    Range: [-1, 1] where 1 means identical direction

    Args:
        vec1, vec2: numpy arrays of same dimension

    Returns:
        Float cosine similarity score
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def find_nearest_neighbors(word, embeddings, vocab, top_k=5):
    """
    Finds the top-k most similar words to a given word using cosine similarity.

    Computes cosine similarity between the query word's embedding and all
    other word embeddings, then returns the top-k most similar.

    Args:
        word: Query word (string)
        embeddings: Embedding matrix [vocab_size, embed_dim]
        vocab: Vocabulary object
        top_k: Number of nearest neighbors to return

    Returns:
        List of (word, similarity_score) tuples, sorted by similarity
    """
    if word not in vocab.word2idx:
        print(f"  '{word}' not in vocabulary")
        return []

    word_idx = vocab.word2idx[word]
    word_vec = embeddings[word_idx]

    # Compute cosine similarity with all words in vocabulary
    similarities = []
    for idx in range(vocab.vocab_size):
        if idx != word_idx:  # Exclude the word itself
            sim = cosine_similarity(word_vec, embeddings[idx])
            similarities.append((vocab.idx2word[idx], sim))

    # Sort by similarity (highest first) and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def analogy(word_a, word_b, word_c, embeddings, vocab, top_k=5):
    """
    Performs word analogy: A is to B as C is to ?

    Uses the vector arithmetic approach:
        result_vector = embedding(B) - embedding(A) + embedding(C)
    Then finds the word closest to result_vector.

    Example: "UG is to BTech as PG is to ?"
    -> analogy("ug", "btech", "pg", ...) should return "mtech"

    Args:
        word_a, word_b, word_c: Words forming the analogy A:B::C:?
        embeddings: Embedding matrix
        vocab: Vocabulary object
        top_k: Number of results to return

    Returns:
        List of (word, similarity) tuples for the predicted answer
    """
    # Check all words are in vocabulary
    for w in [word_a, word_b, word_c]:
        if w not in vocab.word2idx:
            print(f"  '{w}' not in vocabulary")
            return []

    # Get embedding vectors
    vec_a = embeddings[vocab.word2idx[word_a]]
    vec_b = embeddings[vocab.word2idx[word_b]]
    vec_c = embeddings[vocab.word2idx[word_c]]

    # Compute analogy vector: B - A + C
    result_vec = vec_b - vec_a + vec_c

    # Find nearest words to the result vector (excluding A, B, C)
    exclude_set = {word_a, word_b, word_c}
    similarities = []
    for idx in range(vocab.vocab_size):
        word = vocab.idx2word[idx]
        if word not in exclude_set:
            sim = cosine_similarity(result_vec, embeddings[idx])
            similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def save_model(embeddings, vocab, losses, model_type, config, output_dir):
    """
    Saves trained embeddings, vocabulary, and training metadata.

    Args:
        embeddings: numpy embedding matrix [vocab_size, embed_dim]
        vocab: Vocabulary object
        losses: List of per-epoch losses
        model_type: "skipgram" or "cbow"
        config: Dictionary of hyperparameters used
        output_dir: Directory to save model files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as numpy file
    emb_path = os.path.join(output_dir, f"{model_type}_embeddings.npy")
    np.save(emb_path, embeddings)

    # Save vocabulary mappings
    vocab_path = os.path.join(output_dir, f"{model_type}_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({
            "word2idx": vocab.word2idx,
            "idx2word": {str(k): v for k, v in vocab.idx2word.items()},
            "vocab_size": vocab.vocab_size,
        }, f)

    # Save training metadata (hyperparameters and loss curve)
    meta_path = os.path.join(output_dir, f"{model_type}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_type": model_type,
            "config": config,
            "losses": losses,
            "vocab_size": vocab.vocab_size,
            "embedding_shape": list(embeddings.shape),
        }, f, indent=2)

    print(f"[INFO] Model saved to {output_dir}/{model_type}_*")


def load_model(model_type, model_dir):
    """
    Loads a previously saved model from disk.

    Args:
        model_type: "skipgram" or "cbow"
        model_dir: Directory containing the saved model files

    Returns:
        Tuple of (embeddings, vocab) where vocab is a Vocabulary-like object
    """
    emb_path = os.path.join(model_dir, f"{model_type}_embeddings.npy")
    vocab_path = os.path.join(model_dir, f"{model_type}_vocab.json")

    embeddings = np.load(emb_path)

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    # Reconstruct a minimal vocab object
    vocab = Vocabulary()
    vocab.word2idx = vocab_data["word2idx"]
    vocab.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
    vocab.vocab_size = vocab_data["vocab_size"]

    return embeddings, vocab


# ============================================================
# MAIN: TRAINING WITH HYPERPARAMETER EXPERIMENTS
# ============================================================
def load_sentences():
    """
    Loads the preprocessed sentences from the corpus file.

    Returns:
        List of lists of tokens (each inner list is a sentence)
    """
    sentences_path = os.path.join(PROCESSED_DATA_DIR, "sentences.txt")
    sentences = []
    with open(sentences_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                sentences.append(tokens)
    print(f"[INFO] Loaded {len(sentences)} sentences")
    return sentences


def run_experiments():
    """
    Runs the full hyperparameter experiment grid as required by the assignment.

    Experiments with:
    1. Embedding dimensions: [50, 100, 200]
    2. Context window sizes: [2, 5, 7]
    3. Number of negative samples: [3, 5, 10]

    Trains both CBOW and Skip-gram for the default configuration,
    then varies one hyperparameter at a time while keeping others fixed.
    """
    sentences = load_sentences()
    if not sentences:
        print("[ERROR] No sentences found. Run preprocess.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ============================================================
    # DEFAULT CONFIGURATION (best balanced settings)
    # ============================================================
    default_config = {
        "embedding_dim": 100,
        "window_size": 5,
        "num_negative": 5,
        "min_count": 3,
        "learning_rate": 0.001,
        "epochs": 15,
        "batch_size": 512,
    }

    all_results = {}

    # ============================================================
    # EXPERIMENT 1: Vary embedding dimension
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Varying Embedding Dimension")
    print("=" * 60)
    for embed_dim in [50, 100, 200]:
        for model_type in ["skipgram", "cbow"]:
            config = {**default_config, "embedding_dim": embed_dim}
            key = f"{model_type}_dim{embed_dim}"

            embeddings, vocab, losses = train_word2vec(
                sentences,
                model_type=model_type,
                embedding_dim=embed_dim,
                window_size=config["window_size"],
                num_negative=config["num_negative"],
                min_count=config["min_count"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                device=device,
            )

            if embeddings is not None:
                save_dir = os.path.join(MODEL_DIR, key)
                save_model(embeddings, vocab, losses, model_type, config, save_dir)
                all_results[key] = {
                    "final_loss": losses[-1] if losses else None,
                    "config": config,
                }

    # ============================================================
    # EXPERIMENT 2: Vary context window size
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Varying Window Size")
    print("=" * 60)
    for win_size in [2, 5, 7]:
        for model_type in ["skipgram", "cbow"]:
            config = {**default_config, "window_size": win_size}
            key = f"{model_type}_win{win_size}"

            # Skip if already trained with default config
            if key in all_results:
                continue

            embeddings, vocab, losses = train_word2vec(
                sentences,
                model_type=model_type,
                embedding_dim=config["embedding_dim"],
                window_size=win_size,
                num_negative=config["num_negative"],
                min_count=config["min_count"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                device=device,
            )

            if embeddings is not None:
                save_dir = os.path.join(MODEL_DIR, key)
                save_model(embeddings, vocab, losses, model_type, config, save_dir)
                all_results[key] = {
                    "final_loss": losses[-1] if losses else None,
                    "config": config,
                }

    # ============================================================
    # EXPERIMENT 3: Vary number of negative samples
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Varying Negative Samples")
    print("=" * 60)
    for num_neg in [3, 5, 10]:
        for model_type in ["skipgram", "cbow"]:
            config = {**default_config, "num_negative": num_neg}
            key = f"{model_type}_neg{num_neg}"

            if key in all_results:
                continue

            embeddings, vocab, losses = train_word2vec(
                sentences,
                model_type=model_type,
                embedding_dim=config["embedding_dim"],
                window_size=config["window_size"],
                num_negative=num_neg,
                min_count=config["min_count"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                device=device,
            )

            if embeddings is not None:
                save_dir = os.path.join(MODEL_DIR, key)
                save_model(embeddings, vocab, losses, model_type, config, save_dir)
                all_results[key] = {
                    "final_loss": losses[-1] if losses else None,
                    "config": config,
                }

    # Save all experiment results summary
    results_path = os.path.join(MODEL_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] All experiment results saved to {results_path}")


if __name__ == "__main__":
    run_experiments()
