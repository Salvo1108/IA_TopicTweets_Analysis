from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Read data
#tweets = pd.read_csv(r"../dataset/2020-10/Tweet_2020_10_clean.csv")
#tweets = pd.read_csv(r"../dataset/2020-11/Tweet_2020_11_clean.csv")
#tweets = pd.read_csv(r"../dataset/2020-12/Tweet_2020_12_clean.csv")
#tweets = pd.read_csv(r"../dataset/2021-01/Tweet_2021_01_clean.csv")
#tweets = pd.read_csv(r"../dataset/2021-02/Tweet_2021_02_clean.csv")
#tweets = pd.read_csv(r"../dataset/2021-03/Tweet_2021_03_clean.csv")
#tweets = pd.read_csv(r"../dataset/2021-04/Tweet_2021_04_clean.csv")


comment_words = ''
stopwords = set(STOPWORDS)

# iterate through the csv file
for val in tweets.full_text_processed:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()