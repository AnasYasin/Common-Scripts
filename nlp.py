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
  
  
#_____________________________________________________________________
#Preprocessing Pipeline
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)
 
df["clean_text"] = df["text"].apply(lambda x: clean_text(x))
   
