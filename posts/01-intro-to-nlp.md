---
title: "Intro to NLP"
start_date: 2025-08-07
end_date: "ongoing"
---

![Natural Language Processing Banner](../images/nlp-banner.jpg)

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


## Named-Entity Recognition (NER)

Named-entity recognition is a crucial NLP task that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, time expressions, quantities, and more. This capability is essential for information extraction, question answering, and text summarization.

### Using NLTK for Named-Entity Recognition

NLTK provides built-in functionality for named-entity recognition through its `ne_chunk` and `ne_chunk_sents` functions:

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

article = '\ufeffThe taxi-hailing company Uber brings into very sharp focus the question of whether corporations can be said to have a moral character. If any human being were to behave with the single-minded and ruthless greed of the company, we would consider them sociopathic. Uber wanted to know as much as possible about the people who use its service, and those who don't. It has an arrangement with unroll.me, a company which offered a free service for unsubscribing from junk mail, to buy the contacts unroll.me customers had had with rival taxi companies. Even if their email was notionally anonymised, this use of it was not something the users had bargained for. Beyond that, it keeps track of the phones that have been used to summon its services even after the original owner has sold them, attempting this with Apple's phones even thought it is forbidden by the company.\r\n\r\n\r\nUber has also tweaked its software so that regulatory agencies that the company regarded as hostile would, when they tried to hire a driver, be given false reports about the location of its cars. Uber management booked and then cancelled rides with a rival taxi-hailing company which took their vehicles out of circulation. Uber deny this was the intention. The punishment for this behaviour was negligible. Uber promised not to use this "greyball" software against law enforcement – one wonders what would happen to someone carrying a knife who promised never to stab a policeman with it. Travis Kalanick of Uber got a personal dressing down from Tim Cook, who runs Apple, but the company did not prohibit the use of the app. Too much money was at stake for that.\r\n\r\n\r\nMillions of people around the world value the cheapness and convenience of Uber's rides too much to care about the lack of drivers' rights or pay. Many of the users themselves are not much richer than the drivers. The "sharing economy" encourages the insecure and exploited to exploit others equally insecure to the profit of a tiny clique of billionaires. Silicon Valley's culture seems hostile to humane and democratic values. The outgoing CEO of Yahoo, Marissa Mayer, who is widely judged to have been a failure, is likely to get a $186m payout. This may not be a cause for panic, any more than the previous hero worship should have been a cause for euphoria. Yet there's an urgent political task to tame these companies, to ensure they are punished when they break the law, that they pay their taxes fairly and that they behave responsibly.'

# Tokenize the article into sentences: sentences
sentences = sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)
```

In this example:
1. We first tokenize the article into sentences and then tokenize each sentence into words
2. We apply part-of-speech tagging to each tokenized sentence
3. We use `ne_chunk_sents()` to identify named entities in the tagged sentences
4. We iterate through the chunks and print those labeled as named entities (NE)

### Visualizing Named Entities with Matplotlib

We can visualize the distribution of named entity categories using matplotlib:

```python
from matplotlib import pyplot as plt

# Create the defaultdict: ner_categories
ner_categories = {}

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] = ner_categories.get(chunk.label(), 0) + 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()
```

This code creates a pie chart showing the distribution of different named entity categories in the text.

### Advanced NER with spaCy

spaCy offers more sophisticated named entity recognition capabilities:

```python
# Import spacy
import spacy

# Instantiate the English model: nlp
nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'matcher'])

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)
```

spaCy provides more detailed entity categories than NLTK, including:
- PERSON: People's names
- ORG: Organizations, companies, institutions
- GPE: Geopolitical entities (countries, cities, states)
- DATE: Dates or periods
- MONEY: Monetary values
- And many more

### Multilingual NER with polyglot

For multilingual named entity recognition, polyglot is a powerful library:

```python
from polyglot.text import Text

# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)
    
# Print the type of ent
print(type(ent))

# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)
```

Polyglot is particularly useful when working with non-English text, as it supports named entity recognition in multiple languages.

### Why Named-Entity Recognition Matters

- **Information Extraction**: Helps extract structured information from unstructured text
- **Question Answering**: Enables systems to identify entities that questions are asking about
- **Search Optimization**: Improves search relevance by identifying key entities
- **Content Recommendation**: Helps recommend related content based on shared entities
- **Text Summarization**: Identifies important entities to include in summaries

> **Note**: To use these libraries, you may need to install them first:
> - `pip install spacy` and `python -m spacy download en_core_web_sm`
> - `pip install polyglot` along with its dependencies