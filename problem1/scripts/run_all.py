"""
Master Runner Script for Problem 1
=====================================
Runs the entire pipeline for the Word2Vec assignment:
1. Data collection from IIT Jodhpur sources
2. Text preprocessing and statistics
3. Word2Vec training (scratch + gensim)
4. Semantic analysis
5. Visualization

Usage:
    python run_all.py              # Run everything
    python run_all.py --skip-scrape  # Skip data collection (use existing data)
"""

import sys
import os

# Add scripts directory to path so imports work
sys.path.insert(0, os.path.dirname(__file__))


def main():
    skip_scrape = "--skip-scrape" in sys.argv

    # Step 1: Data Collection
    if not skip_scrape:
        print("\n" + "=" * 70)
        print("STEP 1: DATA COLLECTION")
        print("=" * 70)
        from collect_data import main as collect_main
        collect_main()
    else:
        print("\n[SKIP] Data collection skipped (using existing data)")

    # Step 2: Preprocessing
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING")
    print("=" * 70)
    from preprocess import main as preprocess_main
    preprocess_main()

    # Step 3: Word2Vec Training (from scratch)
    print("\n" + "=" * 70)
    print("STEP 3: WORD2VEC TRAINING (FROM SCRATCH)")
    print("=" * 70)
    from word2vec_scratch import run_experiments
    run_experiments()

    # Step 4: Gensim Comparison
    print("\n" + "=" * 70)
    print("STEP 4: GENSIM WORD2VEC (COMPARISON)")
    print("=" * 70)
    from word2vec_gensim import main as gensim_main
    gensim_main()

    # Step 5: Analysis and Visualization
    print("\n" + "=" * 70)
    print("STEP 5: SEMANTIC ANALYSIS & VISUALIZATION")
    print("=" * 70)
    from analyze_and_visualize import main as analyze_main
    analyze_main()

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE!")
    print("=" * 70)
    print("\nOutputs:")
    print("  - Raw data:       problem1/data/raw/")
    print("  - Processed data: problem1/data/processed/")
    print("  - Models:         problem1/models/")
    print("  - Visualizations: problem1/visualizations/")


if __name__ == "__main__":
    main()
