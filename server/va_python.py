#importing necessary libraries
import zipfile
import copy
import re
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import nltk
import csv
import codecs
from collections import Counter
import ast
import json
from langdetect import detect as langdetect
import progressbar
import sklearn
from sklearn.model_selection import train_test_split
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from keras.layers import Embedding
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
   
   # inputs from user
def ml(tweet_column = 'tweets', labels_column = 'fanboy', languages = ['en'],\
    cleaning_words = ['RT','rt','http','https','www','WWW','al','twitter','co','com','html','unsupportedbrowser',],
    embed_dimension = 300,
    test_size = 0.2,
    num_epochs = 1,
    dataset_name = 'portion.csv'
  ):
  org_data = pd.read_csv(dataset_name)


  #Identifies languages and adds a column as language to the dataset
  class read_languages:
    def __init__(self, dataset):
      self.dataset = dataset

    def read_all(self):
      langs = []
      undetected = []
      dataset = self.dataset
      print(' determining the language ')

      bar = progressbar.ProgressBar(maxval=len(dataset), \
          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
      bar.start();
      for i in range (len(dataset)):
        try :
          pred = langdetect(str(dataset[tweet_column][i]))
          langs.append(pred)
        except :
          undetected.append('could not detect language')
          langs.append(None)
        bar.update(i+1)
      bar.finish()

      dataset['language'] = langs
      self.dataset = dataset

      return(dataset)
    def known_languages(self,langs):
      dataset = self.dataset
      dataset = dataset[dataset['language'].notnull()]
      merged1 = pd.DataFrame()
      for i in langs:
        merged1 = merged1.append(dataset[dataset['language']==i])

      return(merged1)

  # calling the language determiner
  rd_lang = read_languages(org_data)
  rd_lang.read_all()
  output_language = rd_lang.known_languages(languages)

  #Balancing the dataset
  merged = output_language
  non_fan = merged.loc[merged[labels_column] == 0].reset_index(drop=True)
  fan = merged.loc[merged[labels_column] == 1].reset_index(drop=True)
  max_len = min(len(fan),len(non_fan))
  merged = pd.concat([fan[:max_len], non_fan[:max_len]], axis=0)

  #A little pre-processing and removing the stopwords
  tweets = merged
  actual_tweets = tweets[tweet_column].copy()

  lmtzr = WordNetLemmatizer()
  # print('-------Lemmazation--------')
  tweets[tweet_column] = tweets[tweet_column].apply(lambda x: ' '.join([lmtzr.lemmatize(word,'v') for word in x.split() ]))

  ## Iterate over the data to preprocess by removing stopwords
  lines_without_stopwords=[]
  for line in tweets[tweet_column].values:
      line = line.lower()
      line_by_words = re.findall(r'(?:\w+)', line, flags = re.UNICODE) # remove punctuation ans split
      new_line=[]
      additional = cleaning_words
      for word in line_by_words:
        if word not in additional:
            if (len(word)>2 and word not in stop):
                new_line.append(word)
            if(len(word)==2 and word[0].isnumeric()==False and word[1].isnumeric()==False and word not in stop):
                new_line.append(word)
      lines_without_stopwords.append(new_line)
  texts = lines_without_stopwords

  tweets[tweet_column] = texts

  #split the data to train and test
  train_set, test_set, actual_tweets_train, actual_tweets_test = train_test_split(tweets, actual_tweets, test_size=test_size, shuffle=True)

  train_set = train_set.reset_index(drop=True)
  test_set = test_set.reset_index(drop=True)
  actual_tweets_train = actual_tweets_train.reset_index(drop=True)
  actual_tweets_test = actual_tweets_test.reset_index(drop=True)

  embeddings_index = {}
  f = open('glove/glove.6B.%dd.txt' % embed_dimension)
  for line in f:
      values = line.split(' ')
      word = values[0] ## The first entry is the word
      coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
      embeddings_index[word] = coefs
  f.close()

  print("Glove data loaded")

  # USING KERAS TO WORK WITH embeddings


  # For test test set
  # encoder = LabelEncoder()
  # encoder.fit(output_language['fanboy'])
  # encoded_Y = encoder.transform(output_language['fanboy'])
  encoded_Y = test_set[labels_column]
  texts = test_set[tweet_column]

  MAX_NUM_WORDS = 100000
  MAX_SEQUENCE_LENGTH = embed_dimension
  tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=False, split=" ")
  tokenizer.fit_on_texts(texts)
  sequences = tokenizer.texts_to_sequences(texts)
  #sequences = tokenizer.texts_to_matrix(texts, mode='tfidf')

  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))

  data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

  labels_test = to_categorical(np.asarray(encoded_Y))


  #For train train set
  ## Code adapted from (https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)
  # Vectorize the text samples


  # encoder = LabelEncoder()
  # encoder.fit(output_language['fanboy'])
  # encoded_Y = encoder.transform(output_language['fanboy'])
  encoded_Y = train_set[labels_column]
  texts = train_set[tweet_column]

  MAX_NUM_WORDS = 100000
  MAX_SEQUENCE_LENGTH = embed_dimension
  tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=False, split=" ")
  tokenizer.fit_on_texts(texts)
  sequences = tokenizer.texts_to_sequences(texts)
  #sequences = tokenizer.texts_to_matrix(texts, mode='tfidf')

  word_index = tokenizer.word_index
  print('Found %s unique tokens.' % len(word_index))

  data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

  labels = to_categorical(np.asarray(encoded_Y))

  #Split into train and validation
  X_train, X_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.2, shuffle=True)

  ## More code adapted from the keras reference (https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)
  # prepare embedding matrix


  ## EMBEDDING_DIM =  ## seems to need to match the embeddings_index dimension
  EMBEDDING_DIM = embeddings_index.get('a').shape[0]
  num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
  found_words = 0
  not_found = 0
  embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
  for word, i in word_index.items():
      if i > MAX_NUM_WORDS:
          continue
      embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector
          found_words +=1
      else  :
          not_found+1

  # load pre-trained word embeddings into an Embedding layer
  # note that we set trainable = False so as to keep the embeddings fixed
  embedding_layer = Embedding(num_words,
                              EMBEDDING_DIM,
                              embeddings_initializer=Constant(embedding_matrix),
                              input_length=MAX_SEQUENCE_LENGTH,
                              trainable=False)

  ## Code from: https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
  ## To create and visualize a model



  model = Sequential()
  model.add(Embedding(num_words, embed_dimension, input_length=embed_dimension, weights= [embedding_matrix], trainable=False))

  model.add(Dropout(rate = 0.2))
  model.add(Conv1D(128, 2, activation='relu'))
  model.add(MaxPooling1D(pool_size=4))
  model.add(LSTM(embed_dimension))
  model.add(Dense(2, activation='softmax'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Finally training the model
  history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=num_epochs)

  #getting the accuracy score
  score = model.evaluate(data_test, labels_test)
  test_score = score[1]
  print('accuracy is: ',test_score)

  #Get predictions and creating a new dataset of predicted labels including tweets and exporting it to csv
  ynew = model.predict_classes(data_test)
  df = pd.DataFrame({'tweets':test_set[tweet_column],'actual_tweet':actual_tweets_test,'predicted':ynew,'label':test_set[labels_column]})
  # test_dataset.to_csv('predicted_test_set.csv',index=False)


  lst = list()

  for i in range(len(actual_tweets_test)):
      a = {}
      a["tweets"] = actual_tweets_test[i]
      a["fanboy"] = int(ynew[i])
      lst.append(a)

  with open('files/predicted_tweets.json', 'w', encoding='utf-8') as f:
      json.dump(lst, f, ensure_ascii=False, indent=4)
  print("created 'predicted_tweets.json' in 'files'")

  #create json file
  # df = pd.read_csv('predicted_test_set.csv')
  tweets = test_set[tweet_column]
  wordsarray = []
  for i in range(len(tweets)):
    wordsarray += tweets[i]
  c=Counter(wordsarray)
  sorted_d = sorted(c.items(), key=lambda x: x[1], reverse=True)

  lst = list()

  for i in range(len(sorted_d)):
      a = {}
      a["text"] = sorted_d[i][0]
      a["size"] = sorted_d[i][1]
      lst.append(a)

  with open('files/words.json', 'w', encoding='utf-8') as f:
      json.dump(lst, f, ensure_ascii=False, indent=4)
  print("created 'words.json' in 'files'")

  #getting uniq words
  uniq_words = list(set(wordsarray))
  #making a csv containing each word with its label
  words=[]
  relateds = []
  fanboys = []
  fanboy_precentage = []
  word_labels = []
  for i in range(len(uniq_words)):
    related_count=0
    fanboy_count=0
    word = uniq_words[i]
    for j in range(len(df)):
      if word in df[tweet_column][j]:
        if (df['predicted'][j] ==1):
            related_count +=1
        if (df['predicted'][j] ==0):
            fanboy_count +=1
    words.append(word)
    relateds.append(related_count)
    fanboys.append(fanboy_count)
    fanboy_precentage.append(int((fanboy_count/(fanboy_count+related_count))*100))
    if (fanboy_precentage[i]>50):
      word_labels.append(1)
    else:
      word_labels.append(0)
  words_df = pd.DataFrame({'word':words,'related_count':relateds, 'fanboy_count':fanboys, 'fanboy_precentage':fanboy_precentage, 'label':word_labels})

  words_df.to_csv('files/word_predictions.csv',index=False)
  print("created 'word_predictions.csv' in 'files'")
  return(test_score)