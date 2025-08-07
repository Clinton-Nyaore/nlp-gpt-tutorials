---
title: "Intro to NLP"
date: 2025-08-07
---

# Intro to NLP

In this learning session, I explored the basics of Natural Language Processing (NLP) with a focus on regular expressions, tokenization using NLTK, and visualizing word lengths in a script. Below is a summary of what I learned and practiced.

---

## Regular Expressions & Word Tokenization

### Introduction to Regular Expressions

I used Python’s `re` module to practice various regular expression operations:

- **Extracting words**:  
  ```python
  re.findall(r"\w+", "Let's write RegEx!")
  ```

- **Splitting sentences** using punctuation like `.?!`:  
  ```python
  re.split(r"[!?]+", sentences)
  ```

- **Finding capitalized words**:  
  ```python
  re.findall(r"[A-Z]\w+", sentences)
  ```

- **Extracting digits**:  
  ```python
  re.findall(r"[0-9]+", sentences)
  ```

---

## Tokenization with NLTK

### Basic Sentence and Word Tokenization

- **Downloaded NLTK’s `punkt` and `punkt_tab`** tokenizer models.
- Used `sent_tokenize()` to split a block of text into sentences.
- Used `word_tokenize()` to tokenize specific sentences into words.

### Pattern Searching in Text

- Searched for keywords (e.g., `"coconuts"`) using `re.search()`.
- Extracted text in square brackets with this pattern:
  ```python
  pattern1 = r"\[.*\]"
  ```

---

## Advanced Tokenization with NLTK and Regex

### Tokenizing Tweets

- Explored `regexp_tokenize()` to extract **hashtags and mentions** from tweets:
  ```python
  pattern = r"[#@]\w+"
  ```

### Tokenizing Multilingual and Emoji Text

- Tokenized a German sentence containing emoji.
- Extracted:
  - All words
  - Capitalized words using `"[A-ZÜ]\w+"`
  - Emoji characters using Unicode ranges

---

## Visualizing Word Length with Matplotlib

### Analyzing Script Line Lengths

- Cleaned speaker tags from lines in a script using:
  ```python
  pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
  ```

- Tokenized each line using `regexp_tokenize()`.
- Counted the number of words per line.
- Plotted a histogram to show the distribution of line lengths:
  ```python
  plt.hist(line_num_words)
  plt.show()
  ```

---

## Final Thoughts

This session gave me a solid foundation in:
- Writing and applying regular expressions for text parsing.
- Tokenizing English and multilingual texts using NLTK.
- Extracting structured patterns (hashtags, mentions, emoji).
- Visualizing linguistic features using Python.

This is an essential first step toward mastering NLP!
