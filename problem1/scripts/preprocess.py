"""
Text Preprocessing Pipeline for IIT Jodhpur Corpus
====================================================
This script takes the raw scraped text files and produces a clean corpus
suitable for Word2Vec training. It also computes dataset statistics and
generates a Word Cloud visualization.

Preprocessing steps (as required by the assignment):
1. Removal of boilerplate text and formatting artifacts
2. Tokenization (using NLTK word tokenizer)
3. Lowercasing
4. Removal of excessive punctuation and non-textual content
5. Removal of stopwords and very short tokens
"""

import os
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ============================================================
# CONFIGURATION
# ============================================================
# Directory paths relative to the project structure
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")

# Minimum token length to keep (very short tokens are usually noise)
MIN_TOKEN_LENGTH = 2

# Common web boilerplate words that add no semantic value
BOILERPLATE_WORDS = {
    "click", "here", "read", "more", "menu", "home", "skip",
    "content", "navigation", "search", "login", "logout", "cookie",
    "cookies", "accept", "privacy", "policy", "terms", "conditions",
    "copyright", "reserved", "rights", "website", "page", "site",
    "www", "http", "https", "html", "css", "javascript", "pdf",
    "download", "upload", "email", "phone", "fax", "tel",
}


def load_raw_documents():
    """
    Loads all raw text files from the data/raw/ directory.

    Returns:
        List of (filename, text_content) tuples
    """
    documents = []
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[ERROR] Raw data directory not found: {RAW_DATA_DIR}")
        print("Please run collect_data.py first.")
        return documents

    for filename in sorted(os.listdir(RAW_DATA_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(RAW_DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append((filename, text))

    print(f"[INFO] Loaded {len(documents)} raw documents")
    return documents


def clean_text(text):
    """
    Cleans a single text document by removing boilerplate and formatting artifacts.

    Processing steps:
    1. Remove URLs and email addresses
    2. Remove HTML entities and special characters
    3. Remove numbers-only tokens (but keep alphanumeric like 'B.Tech')
    4. Normalize whitespace

    Args:
        text: Raw text string

    Returns:
        Cleaned text string (still untokenized)
    """
    # Remove URLs (http/https/www patterns)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)

    # Remove HTML entities like &amp; &nbsp; etc.
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)

    # Remove standalone numbers (but keep alphanumeric tokens like "CSL7640")
    text = re.sub(r'\b\d+\b', ' ', text)

    # Remove excessive punctuation (keep periods and hyphens for abbreviations)
    text = re.sub(r'[^\w\s\.\-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_and_filter(text, stop_words):
    """
    Tokenizes text and applies filtering to produce clean tokens.

    Steps:
    1. Lowercase the entire text
    2. Tokenize using NLTK's word_tokenize (handles contractions, abbreviations)
    3. Remove stopwords (common English words with little semantic meaning)
    4. Remove boilerplate web words
    5. Remove tokens shorter than MIN_TOKEN_LENGTH
    6. Remove tokens that are purely punctuation or numbers

    Args:
        text: Cleaned text string
        stop_words: Set of stopwords to remove

    Returns:
        List of clean tokens
    """
    # Lowercase for normalization
    text = text.lower()

    # Tokenize using NLTK (better than simple split for handling punctuation)
    tokens = word_tokenize(text)

    # Filter tokens based on multiple criteria
    filtered_tokens = []
    for token in tokens:
        # Skip tokens shorter than minimum length
        if len(token) < MIN_TOKEN_LENGTH:
            continue

        # Skip stopwords (common words like 'the', 'is', 'at', etc.)
        if token in stop_words:
            continue

        # Skip boilerplate web-related words
        if token in BOILERPLATE_WORDS:
            continue

        # Skip tokens that are purely punctuation or numbers
        if not re.search(r'[a-zA-Z]', token):
            continue

        # Remove leading/trailing punctuation from tokens
        token = token.strip('.-_')
        if len(token) >= MIN_TOKEN_LENGTH:
            filtered_tokens.append(token)

    return filtered_tokens


def compute_statistics(all_tokens, documents):
    """
    Computes and prints dataset statistics as required by the assignment.

    Reports:
    - Total number of documents
    - Total number of tokens
    - Vocabulary size (unique tokens)
    - Top 20 most frequent words
    - Average document length

    Args:
        all_tokens: List of all tokens in the corpus
        documents: List of (filename, tokens) tuples for per-document stats

    Returns:
        Dictionary containing all computed statistics
    """
    total_tokens = len(all_tokens)
    vocab = set(all_tokens)
    vocab_size = len(vocab)
    word_freq = Counter(all_tokens)

    stats = {
        "total_documents": len(documents),
        "total_tokens": total_tokens,
        "vocabulary_size": vocab_size,
        "avg_tokens_per_doc": total_tokens / max(len(documents), 1),
        "top_20_words": word_freq.most_common(20),
    }

    # Print statistics in a formatted manner
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total documents:        {stats['total_documents']}")
    print(f"Total tokens:           {stats['total_tokens']:,}")
    print(f"Vocabulary size:        {stats['vocabulary_size']:,}")
    print(f"Avg tokens per doc:     {stats['avg_tokens_per_doc']:.1f}")
    print(f"\nTop 20 most frequent words:")
    for word, count in stats["top_20_words"]:
        print(f"  {word:20s} : {count:,}")

    return stats


def generate_wordcloud(word_freq, output_path):
    """
    Generates and saves a Word Cloud visualization of the most frequent words.
    Uses the word frequency distribution to size words proportionally.

    Args:
        word_freq: Counter object with word frequencies
        output_path: Path to save the word cloud image
    """
    # Create word cloud with IIT Jodhpur-relevant styling
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=150,
        colormap="viridis",
        contour_width=1,
        contour_color="steelblue",
        min_font_size=8,
    )

    # Generate from frequency dictionary
    wc.generate_from_frequencies(dict(word_freq.most_common(300)))

    # Save the figure
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title("Word Cloud - IIT Jodhpur Corpus", fontsize=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Word cloud saved to {output_path}")


def main():
    """
    Main preprocessing pipeline. Orchestrates:
    1. Loading raw documents
    2. Cleaning and tokenizing each document
    3. Computing dataset statistics
    4. Generating word cloud
    5. Saving the cleaned corpus and per-document tokenized files
    """
    print("=" * 60)
    print("Text Preprocessing Pipeline")
    print("=" * 60)

    # Load English stopwords from NLTK
    stop_words = set(stopwords.words("english"))

    # Load raw documents
    documents = load_raw_documents()
    if not documents:
        return

    # Process each document: clean, tokenize, and filter
    processed_docs = []
    all_tokens = []

    for filename, text in documents:
        # Step 1: Clean the raw text (remove URLs, artifacts, etc.)
        cleaned = clean_text(text)

        # Step 2: Tokenize and filter (lowercase, remove stopwords, etc.)
        tokens = tokenize_and_filter(cleaned, stop_words)

        if tokens:
            processed_docs.append((filename, tokens))
            all_tokens.extend(tokens)

    print(f"[INFO] Processed {len(processed_docs)} documents (non-empty)")

    # Compute and display dataset statistics
    word_freq = Counter(all_tokens)
    stats = compute_statistics(all_tokens, processed_docs)

    # Create output directories
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # Save the full cleaned corpus as a single file (one sentence per line)
    # Each document's tokens are joined into a single line
    corpus_path = os.path.join(PROCESSED_DATA_DIR, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for filename, tokens in processed_docs:
            # Write each document as one line of space-separated tokens
            f.write(" ".join(tokens) + "\n")
    print(f"[INFO] Corpus saved to {corpus_path}")

    # Also save as sentences (split long documents into ~50 token chunks)
    # This helps Word2Vec learn better context since very long sequences
    # can dilute the local context window
    sentences_path = os.path.join(PROCESSED_DATA_DIR, "sentences.txt")
    chunk_size = 50  # tokens per sentence chunk
    with open(sentences_path, "w", encoding="utf-8") as f:
        for filename, tokens in processed_docs:
            # Split document tokens into chunks to form "sentences"
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i + chunk_size]
                if len(chunk) >= 5:  # Only keep chunks with at least 5 tokens
                    f.write(" ".join(chunk) + "\n")
    print(f"[INFO] Sentences file saved to {sentences_path}")

    # Save statistics as JSON for use in the report
    stats_serializable = {
        "total_documents": stats["total_documents"],
        "total_tokens": stats["total_tokens"],
        "vocabulary_size": stats["vocabulary_size"],
        "avg_tokens_per_doc": round(stats["avg_tokens_per_doc"], 1),
        "top_20_words": stats["top_20_words"],
    }
    stats_path = os.path.join(PROCESSED_DATA_DIR, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, indent=2)

    # Generate word cloud visualization
    wordcloud_path = os.path.join(VIS_DIR, "wordcloud.png")
    generate_wordcloud(word_freq, wordcloud_path)

    print("\n[DONE] Preprocessing complete!")


if __name__ == "__main__":
    main()
