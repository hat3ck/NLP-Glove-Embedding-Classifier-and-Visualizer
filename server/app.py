from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime
from flask_cors import CORS, cross_origin

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import json
import va_python
import download_embedding

app = Flask(__name__)
cors = CORS(app)


@app.route('/getdata', methods=['GET'])
@cross_origin()
def get_data():
    selected_dataset = str(request.args.get('selected_dataset'))
    selected_embed = int(request.args.get('selected_embed'))
    tweets_column = str(request.args.get('tweets_column'))
    labels_column = str(request.args.get('labels_column'))
    test_size = float(request.args.get('test_size'))
    num_epochs = int(request.args.get('num_epochs'))
    lang_selected = np.array(request.args.get('lang_selected').split(","))
    cleaning_words = np.array(request.args.get('cleaning_words').split(","))
    print(f'Received dataset correctly')
   #  downloading the word embeddings
    download_embedding.download_glove()
   #  Running machine learning algorithm
    accuracy = va_python.ml(tweet_column = tweets_column, labels_column = labels_column, languages = lang_selected,\
    cleaning_words = cleaning_words,
    embed_dimension = selected_embed,
    test_size = test_size,
    num_epochs = num_epochs,
    dataset_name = selected_dataset)
    return jsonify(accuracy)



# va_python.ml()


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)
