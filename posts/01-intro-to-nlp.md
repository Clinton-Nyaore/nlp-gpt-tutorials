---
title: "Intro to NLP"
start_date: 2025-08-07
end_date: "ongoing"
---

# Intro to NLP

In this learning session, we explored the basics of Natural Language Processing (NLP) with a focus on regular expressions, tokenization using NLTK, and visualizing word lengths in a script. Below is a summary of what we learned and practiced.

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


## Text Processing and Feature Extraction (NLTK + Gensim)

### Bag of Words from Raw Text

- Tokenize, lowercase, and build a simple bag-of-words (BoW):
  ```python
  from nltk.tokenize import word_tokenize
  from collections import Counter

  article = """'Debugging' is the process of finding and resolving defects that prevent correct operation of computer software or a system.

  Numerous books have been written about debugging (see below: #Further reading|Further reading), as it involves numerous aspects, 
  including interactive debugging, control flow, integration testing, Logfile|log files, monitoring (Application monitoring|application, 
  System Monitoring|system), memory dumps, Profiling (computer programming)|profiling, Statistical Process Control, 
  and special design tactics to improve detection while simplifying changes.

  Origin
  A computer log entry"""

  tokens = word_tokenize(article)
  lower_tokens = [t.lower() for t in tokens]

  bow_simple = Counter(lower_tokens)
  bow_simple.most_common(10)
  ```

### Cleaning, Stopwords, and Lemmatization

- Keep alphabetic tokens, remove stopwords, lemmatize, and rebuild BoW:
  ```python
  # Retain alphabetic words
  alpha_only = [t for t in lower_tokens if t.isalpha()]

  # Remove English stopwords
  from nltk.corpus import stopwords
  no_stops = [t for t in alpha_only if t not in stopwords.words('english')]

  # Lemmatize tokens
  from nltk.stem import WordNetLemmatizer
  wordnet_lemmatizer = WordNetLemmatizer()
  lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

  # Final BoW
  from collections import Counter
  bow = Counter(lemmatized)
  print(bow.most_common(10))
  ```

> Tip: Ensure NLTK data is available (e.g., `punkt`, `stopwords`, `wordnet`).
> You can download it using `nltk.download("punkt")`, `nltk.download("stopwords")`, and `nltk.download("wordnet")`.

### Dictionary, Corpus, and TF–IDF with Gensim

- Build a `Dictionary` and BoW `corpus` from tokenized articles, then compute TF–IDF:
  ```python
  # articles: list of tokenized documents, e.g., [["debugging", "process", ...], ...]
  from gensim.corpora.dictionary import Dictionary
  from gensim.models.tfidfmodel import TfidfModel

  dictionary = Dictionary(articles)

  # Look up a token's id and back
  computer_id = dictionary.token2id.get("computer")
  print(dictionary.get(computer_id))

  # Create BoW corpus
  corpus = [dictionary.doc2bow(article) for article in articles]
  print(corpus[4][:10])  # first 10 (token_id, count) pairs from 5th doc

  # TF–IDF model and top-weighted terms for a document
  tfidf = TfidfModel(corpus)
  doc = corpus[0]  # choose a document from the corpus
  tfidf_weights = tfidf[doc]
  print(tfidf_weights[:5])

  # Sort and view top 5
  sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
  for term_id, weight in sorted_tfidf_weights[:5]:
      print(dictionary.get(term_id), weight)
  ```

#### TF–IDF Formula and Why It Matters

- Term Frequency (TF): frequency of term t in document d
  - TF(t, d) = count(t in d) / total_terms(d)
- Inverse Document Frequency (IDF): down-weights terms common across many documents
  - IDF(t) = log((N + 1) / (df(t) + 1)) + 1, where N = number of documents, df(t) = docs containing t
- TF–IDF score:
  - TF–IDF(t, d) = TF(t, d) × IDF(t)

Why it matters:
- Emphasizes terms that are important for a document but not ubiquitous across the corpus.
- Helps improve retrieval, document ranking, and feature quality for classic ML models.
- Acts as a strong baseline representation before moving to embeddings or transformers.

---

## Regular Expressions & Word Tokenization

### Introduction to Regular Expressions

We used Python’s `re` module to practice various regular expression operations:

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

