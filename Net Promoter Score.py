#!/usr/bin/env python
# coding: utf-8

# # Net Promoter Score Evaluation:
# This model showcases how to evaluate customer satisfaction from online reviews at scale. One can identify information about the market's sentiment for a product or service from survey results. 
# 
# Process: a product was selected at random from those containing over 75 reviews(Amazon's dataset). Reviews were then segmented for investigation by sentiment analysis score. Exploring the review score's top and bottom tiers, clear insights emerge for actionable improvements.

# In[1]:


import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk
import spacy

#Data for this project can also be stored in MongoDB on AWS EC2 and accessed using the PyMongo module
#from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora, models, similarities, matutils
from gensim.corpora import Dictionary
import gensim

from corextopic import corextopic as ct
from corextopic import vis_topic as vt
import scipy.sparse as ss

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Import:
# This data is readily available on Amazon reviews. For this project, the Electronics dataset is used but you can use any data which encopasses similar user sentiments. The following code extracts a user-defined number of records from the electronics reviews collection, scans for null values, and creates a new dataframe for the columns of interest. 
# 
# Reviews are scaled to Net Promoter Scores with the following assumptions: 5 stars equates to a promoter, 4 stars is neutral, and 1-3 stars is a detractor.

# In[2]:


#Making sure that we run the models on the desktop GPUs (RTX 2080Ti). If not, the models will default to using the CPU. 
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import tensorflow.compat.v1 as tfc
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tfc.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# In[3]:


fields = ['product_id', 'product_title', 'review_body', 'review_headline', 
          'star_rating', 'total_votes', 'verified_purchase', 'vine']
df = pd.DataFrame(dtype=np.float64)

'''Faced an issue with allocating huge arrays in Pandas. The kernel was obviously not ready to commit that many physical 
pages to this, and it refuses the allocation. Hence, addressed that by passing nrows. 
Not required if RAM isn't a bottleneck. Also, can use Dask.'''
df=pd.read_csv('..\\speech\\amazon_reviews_us_Electronics_v1_00.tsv', nrows = 90000, sep='\t', usecols=fields)




#df = df[['product_id', 'product_title', 'review_body', 'review_headline', 'star_rating', 'total_votes', 'verified_purchase', 'vine']]

# Assign 5 star reviews to 1, 4 star reviews to 0, and 1-3 star reviews to -1:
df['nps'] = np.where(df['star_rating'] == 5, 1, np.where(df['star_rating'] == 4, 0, -1))
df.to_pickle('..\\speech\\df_records.pkl') 
df.head()


# In[4]:


# create dataframe of products with > 75 reviews:
df_counts = df.groupby(['product_id']).count().sort_values('product_title', ascending = False)
sort_mask = (df_counts['product_title'] > 75)

df_counts[sort_mask].head()


# In[5]:


# for this example, the 3rd product was "randomly" selected:
product_mask = (df['product_id'] == 'B00F5NE2KG')
df_product = df[product_mask]
df_product.head()


# # Exploring Data:
# Online reviews are notably negatively skewed(referring to the fact that the left side of the distribution is the longer “tail”). Frequently, the only customers willing to put this much time into leaving a rating or review are those that have a strong opinion of your product (often translating into a 1-star or 5-star review). As a derivative of star ratings, Net Promoter Score follows a similar pattern. With both medians also equal to the highest rating, we can hypothesize that there is signal loss in simplified ratings systems.

# In[6]:


df_product.describe()


# In[7]:


df_product.star_rating.hist()

print("Mean Star Rating:", df_product.star_rating.mean())
print("Median Star Rating:", df_product.star_rating.median())


# In[8]:


df_product.nps.hist()

print("Mean NPS Rating:", df_product.nps.mean())
print("Median NPS Rating:", df_product.nps.median())


# # Text Pre-processing with Spacy:
# The text data are processed with spaCy and loaded to a new column. The functions alphanumeric and punc_lower remove unwanted punctuation and convert text to lowercase, respectively.

# In[9]:


#nltk.download()
nlp = spacy.load("C:\\Users\\bhise\\Anaconda3\\envs\\speech\\lib\\site-packages\\en_core_web_sm\\en_core_web_sm-2.3.1")

stop_words = spacy.lang.en.stop_words.STOP_WORDS
stop_words.update(['-PRON-', 'pron', 'br', '<br/>', '<br />', '<br>', '<br', 'br>', '</br', 'br/>', '<br /><br />', 'use', 'like', '/br', 'br/'])

# Prepare data to process with Spacy:

list_of_spacy = []

for document in nlp.pipe(df_product.review_body, n_threads = -1):
    try:
        list_of_spacy.append(document)
        
    except:
        list_of_spacy.append('bad_doc')
        print('bad doc:', document)
    
df_product['review_body_spacy'] = list_of_spacy

print(f'{len(list_of_spacy)} documents processed with Spacy\n')
df_product.to_pickle('..\\speech\\df_product-spacy.pkl') 

# pre-process text with NLTK for Corex:

alphanumeric = lambda x: re.sub(r'<[^>]*>\w*\d\w*', '', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x.lower())

words = set(nltk.corpus.words.words())

nltk_list = []

df_product['cleaned_review_body_text'] = df_product['review_body_spacy'].apply(lambda x: ' '.join([ele.lemma_ for ele in x if ele.lemma_ not in stop_words and ele not in stop_words]))

for row in df_product['cleaned_review_body_text']:
    nltk_list.append(' '.join(w for w in nltk.wordpunct_tokenize(row) if w.lower() in words or not w.isalpha()))
    
df_product['nltk_terms'] = nltk_list
df_product['nltk_terms'] = df_product['nltk_terms'].map(alphanumeric).map(punc_lower)
df_product = df_product.drop(columns = ['cleaned_review_body_text'])

print('NLTK processing complete.')
df_product.to_pickle('..\\speech\\df_product-nltk.pkl') 
df_product.head()


# # NLP Hypothesis Testing:
# NLP models can help find the accuracy in the language and find the true meanings behind them. The nuances between "Bloody" and "Bloody good" have to be identified and contenxualized. Before diving into a protracted NLP analysis to find where the ratings have disappointed, it is important to determine whether a general correlation between product reviews and ratings exists.
# 
# This will be accomplished by applying classification to a Bag-of-Words model, specifically using CountVectorizer and Logistic Regression.

# In[10]:


# create a new dataframe with review body text and nps:

df2 = df[['review_body', 'nps']]
df2 = df2.drop(df2[df2.nps == 0].index) # remove neutral rating (4 stars) for binary classification
df2.head()


# In[11]:


# define features & labels:

X = df2.review_body.values.astype('unicode')
y = df2.nps

# split into training / test sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# document-term matrix for default Count Vectorizer values - counts of unigrams:

cv1 = CountVectorizer(max_features=20000, stop_words='english', token_pattern="\\b[a-z][a-z]+\\b", binary=True)

X_train_cv1 = cv1.fit_transform(X_train)
X_test_cv1  = cv1.transform(X_test)

pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names()).head()


# In[12]:


# document-term matrix has both unigrams and bigrams, and indicators instead of counts:

cv2 = CountVectorizer(max_features=20000, stop_words='english', ngram_range=(1,2), token_pattern="\\b[a-z][a-z]+\\b", binary=True)
                    
X_train_cv2 = cv2.fit_transform(X_train)
X_test_cv2  = cv2.transform(X_test)

pd.DataFrame(X_train_cv2.toarray(), columns=cv2.get_feature_names()).head()


# In[13]:


# Create a logistic regression model to use
lr = LogisticRegression()

# Train the first model
lr.fit(X_train_cv1, y_train)
y_pred_cv1 = lr.predict(X_test_cv1)

# Train the second model
lr.fit(X_train_cv2, y_train)
y_pred_cv2 = lr.predict(X_test_cv2)


# In[14]:


# Create a function to calculate the error metrics, since we'll be doing this several times

def conf_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'], 
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu");

    true_neg, false_pos = cm[0]
    false_neg, true_pos = cm[1]

    accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg),3)
    precision = round((true_pos) / (true_pos + false_pos),3)
    recall = round((true_pos) / (true_pos + false_neg),3)
    f1 = round(2 * (precision * recall) / (precision + recall),3)

    cm_results = [accuracy, precision, recall, f1]
    return cm_results


# # Results:
# The supervised learning model confirms a strong connection between reviews and ratings with all metrics above 90% and F1 scores nearing 94%.

# In[15]:


# Heat map for the first logistic regression model:

cm1 = conf_matrix(y_test, y_pred_cv1)


# In[16]:


# Heat map for the second logistic regression model

cm2 = conf_matrix(y_test, y_pred_cv2)


# In[17]:


# Compare results:

results = pd.DataFrame(list(zip(cm1, cm2)))
results = results.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results.columns = ['LogReg1', 'LogReg2']
results


# # VADER Sentiment Analysis:
# The star ratings & NPS data's extreme left skew showcases a bias for 5 Star Ratings. Are these manipulated numbers? How then does a product discern between those 5 Star Ratings from lethargic or generous customers and those reviews indicative of raving fans? 
# 
# This skewness wouldn’t be a problem if it weren’t for the fact that we are hardwired to see shapes as symmetrical around a clearly defined central tendency (a phenomenon known as the Law of Symmetry in Gestalt psychology). 
# 
# The most enthusiastic fans were isolated from the "5-Star by Default" population using Valence Aware Dictionary and sEntiment Reasoner (VADER) Sentiment Analysis. A filter coupling VADER scores over 0.95 with an NPS Score of 1 (as a checksum for sarcasm) was then used to populate a dataframe comprised of the most satisfied customers. This process was then repeated with inverse parameters to isolate the least satisfied customers.
# 
# If this step is missed, a 5-star review can easily trick a Product Manager into thinking that they have build an exceptional, customer-first app or service, even though they really only have an average rating.

# In[18]:



# use VADER with original reviews column as input to include emotionally rich punctuation and capitalization: 

analyzer = SentimentIntensityAnalyzer()

# compound is the normalized, weighted composite score:

df_product['vader'] = [analyzer.polarity_scores(row)['compound'] for row in df_product.review_body] 

print(f'created VADER scores for {df_product.vader.shape[0]} rows\n')

df_product.head()


# In[19]:


df_product.vader.hist()


# In[20]:


df_product.corr()


# In[21]:


# explore top reviews using a vader filter > 0.95:

df_p1_top_mask = (df_product['vader'] > 0.95)

df_product[df_p1_top_mask].nps.hist()


# In[22]:


df_product[df_p1_top_mask].star_rating.hist()


# In[23]:


# use a combination of vader and nps to create dataframe for top and bottom reviews:

df_p1_top_mask = (df_product['vader'] > 0.95) & (df_product['nps'] == 1)
df_top = df_product[df_p1_top_mask]

df_p1_bot_mask = (df_product['vader'] < -0.25) & (df_product['nps'] == -1)
df_bot = df_product[df_p1_bot_mask]


# In[24]:



cor_vectorizer = CountVectorizer(max_features=20000, ngram_range=(1,2), binary=True, token_pattern="\\b[a-z][a-z]+\\b", stop_words='english')

cor_doc_word_top = cor_vectorizer.fit_transform(df_top['nltk_terms'])
cor_words = list(np.asarray(cor_vectorizer.get_feature_names())) 
topic_model_top = ct.Corex(n_hidden=6, words = cor_words, seed=1)
topic_model_top.fit(cor_doc_word_top, words = cor_words, docs = df_top.nltk_terms)

# repeat process for bottom reveiws:

cor_doc_word_bot = cor_vectorizer.fit_transform(df_bot['nltk_terms'])
cor_words = list(np.asarray(cor_vectorizer.get_feature_names()))
topic_model_bot = ct.Corex(n_hidden=6, words = cor_words, seed=1) # must be repeated
topic_model_bot.fit(cor_doc_word_bot, words = cor_words, docs = df_bot.nltk_terms)


# In[25]:


# Print all topics from the top topic model:

topics = topic_model_top.get_topics()
for n, topic in enumerate(topics):
    if topic:
        topic_words,_ = zip(*topic)
        print('{}: '.format(n) + ','.join(topic_words))


# In[26]:


# retrieve original reviews & top topics for parameters specified:

def TopicReviews(topic_number, topic_model_version, topic_dataframe):
    
    print('Topics: \n\n', topic_model_version.get_topics()[topic_number])

    top_docs = topic_model_version.get_top_docs(topic=topic_number, n_docs=3)

    for i in top_docs:
        temp = (topic_dataframe.loc[topic_dataframe['nltk_terms'] == i[0]]).index[0]  # get row index of review for original review
        print('\nIndex:', temp)
        print('Rating:', df.star_rating.iloc[temp])
        print('Review:', df.review_body.iloc[temp])  # print original review


# In[27]:


TopicReviews(0, topic_model_top, df_top)


# In[28]:


TopicReviews(1, topic_model_top, df_top)


# In[29]:



plt.figure(figsize=(10,5))
plt.bar(range(topic_model_top.tcs.shape[0]), topic_model_top.tcs, color='#ffa500', width=0.5)
plt.title('Correlation by Topic: Top Reviews')
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16);


# In[30]:


# Print all topics from the bottom topic model:

topics = topic_model_bot.get_topics()
for n, topic in enumerate(topics):
    topic_words,_ = zip(*topic)
    print('{}: '.format(n) + ','.join(topic_words))


# In[31]:


TopicReviews(0, topic_model_bot, df_bot)


# In[32]:


TopicReviews(1, topic_model_bot, df_bot)


# In[33]:


plt.figure(figsize=(10,5))
plt.bar(range(topic_model_bot.tcs.shape[0]), topic_model_bot.tcs, color='#4e79a7', width=0.5)
plt.title('Correlation by Topic: Bottom Reviews')
plt.xlabel('Topic', fontsize=16)
plt.ylabel('Total Correlation (nats)', fontsize=16);


# In[ ]:




