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

