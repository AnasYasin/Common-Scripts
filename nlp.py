#Simple Bag of words

from nltk.tokenize import word_tokenize
from collections import Counter
sentance = "the brown fox and the dog"
bow = Counter(word_tokenize(sentance))
print(bow.most_common(2))

#_____________________________________________________________________

#Tokenization, lowercase and stopwords/punctuation removal 
from ntlk.corpus import stopwords
text = "The cat is in the box. The cat likes the box. The box is over the cat."
tokens = [w for w in word_tokenize(text.lower()) if w.isalpha()] # removing puntuations and tokenizing.
no_stops = [t for t in tokens if t not in stopwords.words('english')]
bow = Counter(no_stops)
print(bow.most_common(2))


#_____________________________________________________________________

#lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops] #no_stops is a list of words with lower case and stopwords free words.
# Create the bag-of-words: bow
bow = Counter(lemmatized)
# Print the 10 most common tokens
print(bow.most_common(10))


#_____________________________________________________________________

#Creating and querying a corpus with gensim
from gensim.corpora.dictionary import Dictionary
# Create a Dictionary from the articles: dictionary
dictionary = Dictionary(articles)
# Select the id for "computer": computer_id
computer_id = dictionary.token2id.get("computer")
# Use computer_id with the dictionary to print the word
print(dictionary.get(computer_id))
# Create a MmCorpus: corpus
corpus = [dictionary.doc2bow(article) for article in articles]
# Print the first 10 word ids with their frequency counts from the fifth document
print(corpus[4][:10])

#_____________________________________________________________________

#tf-ifd (grater if the word aperas more in a doc say 50 out wo 100 words plus also aprear more in multiple docs say in 50 out of 100 docs. It will be low if onli apearing more in one doc)

from gensim.models.tfidfmodel import TfidfModel
# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)
# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]
# Print the first five weights
print(tfidf_weights[:5])
# Sort the weights from highest to lowest: sorted_tfidf_weights
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
# Print the top 5 weighted words
for term_id, weight in sorted_tfidf_weights[:5]: 
  print(dictionary.get(term_id), weight)
