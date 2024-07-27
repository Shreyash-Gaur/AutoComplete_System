---

# Auto-Complete System

## Overview

This project demonstrates the development of an auto-complete system using N-gram based language models. Auto-complete systems are widely used in applications such as search engines, email clients, and messaging apps to provide word suggestions that help users complete their sentences or queries.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Functions and Features](#functions-and-features)
7. [Evaluation](#evaluation)
8. [Conclusion](#conclusion)
9. [References](#references)

## Introduction

An auto-complete system predicts the next word or phrase that a user intends to type based on the context of the previously typed text. This project uses Twitter data to build an N-gram based language model that suggests the next word in a sentence.

## Prerequisites

- Python 3.x
- Libraries: `math`, `random`, `numpy`, `pandas`, `nltk`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AutoComplete-System.git
   cd AutoComplete-system
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas nltk
   ```

3. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

1. Load the Twitter dataset (`en_US.twitter.txt`) into the `./data/` directory.
2. Run the script to preprocess data, build the language model, and evaluate it:
   ```bash
   python Auto_Complete.ipynb
   ```

## Methodology

### Data Loading and Preprocessing
1. **Load Data**: Read the Twitter data from a text file.
2. **Preprocess Data**: Clean and tokenize the data, split it into sentences, and further split sentences into words (tokens).

### N-gram Language Model
1. **N-gram Counting**: Count the occurrences of N-grams (e.g., unigrams, bigrams, trigrams) in the training data.
2. **Probability Estimation**: Estimate the conditional probability of each word given its preceding words using K-smoothing.

### Model Evaluation
1. **Perplexity Calculation**: Evaluate the language model by calculating its perplexity on the test data.

### Auto-complete System
1. **Word Suggestion**: Given a sequence of words, suggest the next word that maximizes the probability of the sequence.

## Functions and Features

### Data Preprocessing
- `split_to_sentences(data)`: Splits the raw data into sentences.
- `tokenize_sentences(sentences)`: Tokenizes each sentence into words.
- `get_tokenized_data(data)`: Combines the above functions to return tokenized data.
- `count_words(tokenized_sentences)`: Counts the occurrences of each word in the tokenized data.
- `get_words_with_nplus_frequency(tokenized_sentences, count_threshold)`: Gets words with a frequency above a specified threshold.
- `replace_oov_words_by_unk(tokenized_sentences, vocabulary)`: Replaces words not in the vocabulary with `<unk>`.

### N-gram Language Model
- `count_n_grams(tokenized_sentences, n)`: Counts the occurrences of N-grams.
- `estimate_probability(word, previous_n_gram, n_gram_counts, vocabulary_size, k)`: Estimates the probability of a word given the previous N-gram using K-smoothing.

### Perplexity Calculation
- `calculate_perplexity(test_data, n_gram_counts, n, vocabulary_size, k)`: Calculates the perplexity of the language model on the test data.

### Auto-complete System
- `suggest_a_word(previous_words, n_gram_counts, vocabulary, k)`: Suggests the next word given a sequence of previous words.

## Evaluation

The N-gram language model is evaluated using the perplexity metric. A lower perplexity indicates a better model, meaning it predicts the test data more accurately.

## Conclusion

This project provides a foundational implementation of an auto-complete system using N-gram language models. While N-grams are simple and effective, more advanced models such as neural networks (e.g., LSTM, Transformer) can be used for better performance.

## References

- [Natural Language Toolkit (NLTK)](https://www.nltk.org/)
- [Kneser-Ney Smoothing](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)
- [Understanding Perplexity](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94)

---
