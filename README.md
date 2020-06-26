# Glove Embeddings Classifier and Visualizer
 This is a platform for text classification and word cloud visualization. It uses python with flash in the backend to perform classification using glove word embeddings with an interface. Also, in the front end it uses the machine learning results to create a word cloud with D3 and JavaScript. 
The details of machine learning preprocessing and Neural Networks can be changed from "va_python" file in the "server" folder. Once the machine learning phase is done it will report the accuracy score of the test set and create the subsequent files inside the files folder and the front-end section will use those files to show the word cloud.  
The word cloud also includes a hovering option which reports the number of occurrence of the word in the whole document and the class that it belongs to (since it is used for binary classification it calculates that which tweets the words belong to and which class does the tweet belong to, as well. In the end, the sum of occurrence of the word in each class would be compared to determine the class of the word). The words are shown in red and blue, representing their class. If you click on any words a window pop up showing the tweets that the word has been used inside them.  
Currently this framework, works with only binary classification text data but with some small changes it could be used for the multi-class classification as well. 
As, a sample Kaggle Isis dataset has been used to illustrate the results.
To use this framework for any other dataset you need to upload the dataset in .csv format inside the server folder and specify the name of the dataset, text column and targets column. You also have the option to choose the test size, epochs for embeddings, language for your tweets(see the langdetect package resources for details of each language's acronym), and the words that you want to exclude from the text (cleaning words).  
Information about how to run the program can be found below:


To run this program first you will need to install the packages
Also, at least 4 GB of space is required to download word embeddings



To use the application:
1- run flask server on backend/app.py with python3 => python3 server/app.py
2- run a simple http_server on main folder => python3 -m http.server
3- open frontend/index.html in a browser => http_server:port_number/frontend/index.html
