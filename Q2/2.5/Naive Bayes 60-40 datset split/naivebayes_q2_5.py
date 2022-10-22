# -*- coding: utf-8 -*-
"""naiveBayes Q2.5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12EMwMLQSSyOw08RUxwNov8vrToriCEtf
"""

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys

file = open(sys.path[0] + '/../../../Q1/data.csv', encoding="utf-8")
performance = open("performance Naive Bayes with 60-40 train test split.txt", "w")
data = pd.read_csv(file, names=['post', 'emotion','sentiment'], header=0)

# set the hyperparameters gridsearchCV
hyperparameters = {
    'alpha': [0, 0.3, 0.5, 0.7], # alpha is for smoothing, best for emotion 0.5, best for sentiment 0.7
                                # setting 0, causes a user warning 
}

# use countvectorizer
vectorizer = CountVectorizer()

# seperate the columns
df_x = data['post']
df_y = data['emotion'].tolist()
df_z = data['sentiment'].tolist()

# get the feature vector of the posts
X = vectorizer.fit_transform(df_x)
performance.write("Naive bayes performance stats")
performance.write('\n\n')
performance.write("2.1 size of vocab from count vectorizer: ")
performance.write( str(len(vectorizer.vocabulary_)))
performance.write('\n')
print("size of vocab from count vectorizer is ", str(len(vectorizer.get_feature_names_out()))) # len of all the words

#_________________________________emotion__________________________________________
x_train, x_test, y_train, y_test= train_test_split(X,df_y, test_size=0.4, random_state=50)
# 2.2 split dataset into training (80%) and testing (20%), random value to keep the set deterministic
classifierEmo = MultinomialNB() # multinomial naive bayes base parameters

# model for default naive bayes and predict
model_emotion = classifierEmo.fit(x_train, y_train)
predictEmoNB = model_emotion.predict(x_test)
print('default Naive bayes emotion accuracy score', accuracy_score(y_test,predictEmoNB))
performance.write('default Naive bayes emotion accuracy score :')
performance.write(str(accuracy_score(y_test,predictEmoNB)))
confusionMatrixEmotion= confusion_matrix(y_test,predictEmoNB)
classificationReportEmotion = classification_report(y_test,predictEmoNB, zero_division=0)#warning occurs when model hasn't predicted a label, hence when making a report causes zero division warning
print(confusionMatrixEmotion)
print(classificationReportEmotion)

performance.write('\n\n classification report of default naive bayes for emotion\n')
performance.write(classificationReportEmotion)
performance.write('\n\n confusion matrix of default naive bayes for emotion\n')
npMatrix = np.array2string(confusionMatrixEmotion)
performance.write(npMatrix)
performance.write('\n\n-------------------------x-------------------------')

# model for naive bayes using the gridsearchCV parameters and predict
grid = GridSearchCV(estimator = classifierEmo, param_grid = hyperparameters) # set the grid search parameters
model_emotion_grid = grid.fit(x_train,y_train)
predictEmoNBCV = model_emotion_grid.predict(x_test)
print('best parameters', grid.best_params_)
print('gridsearchCV Naive bayes emotion accuracy score', accuracy_score(y_test,predictEmoNBCV))
performance.write('\n\ngridsearchCV (alpha value of 0.5) Naive bayes emotion accuracy score :')
performance.write(str(accuracy_score(y_test,predictEmoNBCV)))
confusionMatrixEmotionCV= confusion_matrix(y_test,predictEmoNBCV)
classificationReportEmotionCV = classification_report(y_test,predictEmoNBCV,zero_division=0)
print(confusionMatrixEmotionCV)
print(classificationReportEmotionCV)
performance.write('\n\n classification report of grid search naive bayes (alpha value of 0.5) for emotion\n')
performance.write(classificationReportEmotionCV)
performance.write('\n\n confusion matrix of grid search naive bayes (alpha value of 0.5) naive bayes for emotion\n')
npMatrixCV = np.array2string(confusionMatrixEmotionCV)
performance.write(npMatrixCV)
performance.write('\n\n-------------------------x-------------------------')

#___________________________________sentiment_____________________________________________
x2_train, x2_test, z_train, z_test= train_test_split(X,df_z, test_size=0.4, random_state=50)
classifierse = MultinomialNB()# multinomial naive bayes classifier

# model for default naive bayes and predict
model_sentiment = classifierse.fit(x2_train, z_train)
predictSentNB = model_sentiment.predict(x2_test)
print('default Naive bayes sentiment accuracy score', accuracy_score(z_test,predictSentNB))
performance.write('\n\ndefault Naive bayes sentiment accuracy score :')
performance.write(str(accuracy_score(z_test,predictSentNB)))
confusionMatrixSent= confusion_matrix(z_test,predictSentNB)
classificationReportSent = classification_report(z_test,predictSentNB,zero_division=0)
print(confusionMatrixSent)
print(classificationReportSent)
performance.write('\n\n classification report of default naive bayes for sentiment\n')
performance.write(classificationReportSent)
performance.write('\n\n confusion matrix of default naive bayes for sentiment\n')
npMatrixSent = np.array2string(confusionMatrixSent)
performance.write(npMatrixSent)
performance.write('\n\n-------------------------x-------------------------')

# model for naive bayes using the gridsearchCV parameters and predict
gridse = GridSearchCV(estimator = classifierse, param_grid = hyperparameters)
model_sentiment_CV = gridse.fit(x2_train, z_train)
predictSentNBCV = model_sentiment_CV.predict(x2_test)
print('best parameters', gridse.best_params_)
print('gridsearch Naive bayes sentiment accuracy score', accuracy_score(z_test,predictSentNBCV))
performance.write('\n\ngridsearch (alpha value of 1) Naive bayes sentiment accuracy score :')
performance.write(str(accuracy_score(z_test,predictSentNBCV)))
performance.write('\n')
performance.write('\n')
confusionMatrixSentCV= confusion_matrix(z_test,predictSentNBCV)
classificationReportSentCV = classification_report(z_test,predictSentNBCV,zero_division=0)
print(confusionMatrixSentCV)
print(classificationReportSentCV)
performance.write('\n\n classification report of grid search naive bayes (alpha value of 1) for sentiment\n')
performance.write(classificationReportSentCV)
performance.write('\n\n confusion matrix of grid search naive bayes (alpha value of 1) naive bayes for sentiment\n')
npMatrixSentCV = np.array2string(confusionMatrixSentCV)
performance.write(npMatrixSentCV)
performance.close()
