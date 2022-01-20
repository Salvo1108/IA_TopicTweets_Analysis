# Importing modules
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint



stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])



# Read data
tweets = pd.read_csv('../DataTweet_csv/2020-10-18-23.csv')


# Rimozione di tutte le colonne tranne quella full_text che contiene il testo del tweet
tweets = tweets.drop(tweets.columns.difference(['full_text']), axis=1).sample(5)

# Rimozione punteggiatura
tweets['full_text_processed'] = \
tweets['full_text'].map(lambda x: re.sub('[,\.!?]', '', x))

# Conversione in minuscolo
tweets['full_text_processed'] = \
tweets['full_text_processed'].map(lambda x: x.lower())

# Rimozione menzioni
tweets['full_text_processed'] = \
tweets['full_text_processed'].map(lambda x: re.sub(r'@[A-Za-z0-9]+', '', x))

# Rimozione hyperlink
tweets['full_text_processed'] = \
tweets['full_text_processed'].map(lambda x: re.sub(r'https?:\/\/\S+', '', x))

# Rimozione #
tweets['full_text_processed'] = \
tweets['full_text_processed'].map(lambda x: re.sub(r'#', '',x))

# Rimozione RT
tweets['full_text_processed'] = \
tweets['full_text_processed'].map(lambda x: re.sub(r'RT[\s]+','',x))


# Stampa la colonna nuova creata
tweets['full_text_processed'].head()

# Print head
#print(tweets.head())

# Join the different processed titles together.
long_string = ','.join(list(tweets['full_text_processed'].values))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Visualize the word cloud
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()



def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


data = tweets.full_text_processed.values.tolist()
data_words = list(sent_to_words(data))

#print(data_words[:1][0][:30])

# remove stop words
data_words = remove_stopwords(data_words)


# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1][0][:30])


# number of topics
num_topics = 10

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]