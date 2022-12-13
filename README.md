# Amazon-Reviews-Analysis
Sentiment Analysis of Amazon Product Reviews

Overview
	•	Dataset : The Datafiniti Product Database
	•	Problem : Sentiment analysis for Amazon product reviews
	•	In this project I have tried to perform Binary classification on Amazon reviews by pre-processing the Text and converting it into vectors and then applying different Machine Learning models such as Random Forest Classifier,  and Deep Learning models like Neural Networks, LSTM, BERT Model.The main aim of this experiment is to compare the performance of different ML and DL models.
	•	Accuracy score of 94.4% using Deep Learning Models.
Data preprocessing
Data-Loading : I have loaded a dataset that includes multiple columns such as Review ID, categories, keys, manufacturer, date, and review text.
Data-Preprocessing : Needed only Amazon reviews and ratings for our experiment so sliced the data accordingly.
Data Cleaning : Processed the text by removing unwanted spaces, removing punctuations, converting all the words into lowercase and removing the stopwords found in the English dictionary.
Feature Extraction : Converted the text into a vector using a Tf-IDF vectorizer and a count vectorizer. The features were created by taking all of the words that appear in the corpus and creating tokens out of them.
Code 
Amazon_Review_Analysis.ipynb
	•	Data visualization of Amazon product reviews
	•	Preprocess raw reviews to cleaned reviews
	•	Used different word embedding models, such as count vectorizer,tf-idf transformation and Word2Vec model, to transform text reviews into numerical representations.
	•	Apply ML models : RandomForest Classifier, Multinomial NB, Bernoulli NB, SVM with variations in different parameters and plot the Acurracies.
	•	Apply DL models: Neural Networks, LSTM, BERT.


Results
	•	Two feature representation methods and seven models based on machine learning and deep learning are used for the experiment.
	•	Neural networks were run for an epoch size of 50. LSTM and BERT were run for an epoch size of 3.
	•	The Bert model is the best performer when it comes to accuracy. The machine learning-based Count Vectorizer + Multinomial Naive Bayes model performed better by achieving an accuracy of 93.43%. 
	•	The Neural Networks  is the most successful deep learning approach, with an accuracy  of 94.06%. 
	•	The least performing model was the machine learning-based Tf-Idf Vectorizer + Bernoulli Naive Bayes and Count Vectorizer + Multinomial Naive Bayes model model with an accuracy of 89.58% and 86.87% respectively.

Models
Accuracies(%)

Count Vectorizer
Tf-Idf Vectorizer
Random Forest Classifier
93.21
93.34
Multinomial NB
93.43
93.28
Bernoulli NB
90.19
89.58
SVM
86.87
93.58
Neural Networks
94.06
LSTM
94.23
BERT
94


