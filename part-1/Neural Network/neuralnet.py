# -*- coding: utf-8 -*-
"""neuralNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SSRHrLwKff9sBQvguZM7nezKOpIxfFV_
"""

import numpy as np
from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import sys

file = open(sys.path[0] + '/../../Q1/data.csv', encoding="utf-8")
performance = open("performance Neural Network.txt", "w")

hyperparameters = {
     'activation'     : ['relu','tanh','identity', 'logistic'],
    'hidden_layer_sizes': [[30,50],[10,10,10]],#
    'solver':[ 'adam','sgd'] # 
}
#stochastic not good, doesnt coverge

data = pd.read_csv(file, names=['post', 'emotion','sentiment'], header=0)
vectorizer = CountVectorizer()
le = LabelEncoder()

df_x = data['post']
df_y = data['emotion']
df_z = data['sentiment']
X = vectorizer.fit_transform(df_x)
Y = le.fit_transform(df_y)
Z= le.fit_transform(df_z)

performance.write('Neural Network performance stats\n\n')

#________________________________emotion____________________________________________
x_train, x_test, y_train, y_test= train_test_split(X,Y, test_size=0.2, random_state=50)
neuralEmo = neural_network.MLPClassifier( max_iter=2)
model_nn_emo = neuralEmo.fit(x_train,y_train)
predictEmo = model_nn_emo.predict(x_test)
print('neural network emotion accuracy score', accuracy_score(y_test,predictEmo))
performance.write('default neural network emotion accuracy score :')
performance.write(str(accuracy_score(y_test,predictEmo)))

confusionMatrixEmotion= confusion_matrix(y_test,predictEmo)
classificationReportEmotion = classification_report(y_test,predictEmo, zero_division=0)#warning occurs when model hasn't predicted a label, hence when making a report causes zero division warning
print(confusionMatrixEmotion)
print(classificationReportEmotion)

performance.write('\n\n classification report of default neural network for emotion\n')
performance.write(classificationReportEmotion)
performance.write('\n\n confusion matrix of default neural network for emotion\n')
npMatrix = np.array2string(confusionMatrixEmotion)
performance.write(npMatrix)
performance.write('\n\n-------------------------x-------------------------')

# model for neural network using the gridsearchCV parameters and predict emotion
grid = GridSearchCV(estimator = neuralEmo, param_grid = hyperparameters)
model_nn_emo_cv = grid.fit(x_train,y_train)
predictEmoNNCV = model_nn_emo_cv.predict(x_test)
print('neural network emotion with grid cv accuracy score', accuracy_score(y_test,predictEmoNNCV))
print('neural network emotion with grid cv best parameters ',grid.best_params_)
performance.write('\n\ngridsearch cv neural network emotion accuracy score :')
performance.write(str(accuracy_score(y_test,predictEmoNNCV)))
performance.write('\n\ngridsearch cv neural network emotion best parameters:')
performance.write(str(grid.best_params_))

confusionMatrixEmotionCV= confusion_matrix(y_test,predictEmoNNCV)
classificationReportEmotionCV = classification_report(y_test,predictEmoNNCV, zero_division=0)#warning occurs when model hasn't predicted a label, hence when making a report causes zero division warning
print(confusionMatrixEmotionCV)
print(classificationReportEmotionCV)

performance.write('\n\n classification report of grid search neural network for emotion\n')
performance.write(classificationReportEmotionCV)
performance.write('\n\n confusion matrix of grid search neural network for emotion\n')
npMatrixCV = np.array2string(confusionMatrixEmotionCV)
performance.write(npMatrixCV)
performance.write('\n\n-------------------------x-------------------------')

#_________________________________sentiment_______________________________
x2_train, x2_test, z_train, z_test= train_test_split(X,Z, test_size=0.2, random_state=50)
neuralclasssentiment = neural_network.MLPClassifier( max_iter=2)
model_dec_nn_sent = neuralclasssentiment.fit(x2_train,z_train)
predictsent = model_dec_nn_sent.predict(x2_test)
print('neural network sentiment accuracy score', accuracy_score(z_test,predictsent))
performance.write('\n\ndefault neural network sentiment accuracy score :')
performance.write(str(accuracy_score(z_test,predictsent)))

confusionMatrixSent= confusion_matrix(z_test,predictsent)
classificationReportSent = classification_report(z_test,predictsent, zero_division=0)#warning occurs when model hasn't predicted a label, hence when making a report causes zero division warning
print(confusionMatrixSent)
print(classificationReportSent)

performance.write('\n\n classification report of default neural network for sentiment\n')
performance.write(classificationReportSent)
performance.write('\n\n confusion matrix of default neural network for sentiment\n')
npMatrixSent = np.array2string(confusionMatrixSent)
performance.write(npMatrixSent)
performance.write('\n\n-------------------------x-------------------------')

# model for neural network using the gridsearchCV parameters and predict sentiment
gridse = GridSearchCV(estimator = neuralclasssentiment, param_grid = hyperparameters)
model_nn_sen_cv = gridse.fit(x2_train,z_train)
predictsentNNCV = model_nn_sen_cv.predict(x2_test)
print('neural network sentiment accuracy score', accuracy_score(z_test,predictsentNNCV))
print('neural network sentiment with grid cv best parameters ',gridse.best_params_)
performance.write('\n\ngridsearch cv neural network sentiment accuracy score :')
performance.write(str(accuracy_score(z_test,predictsentNNCV)))
performance.write('\n\ngridsearch cv neural network sentiment best parameters:')
performance.write(str(gridse.best_params_))

confusionMatrixSentCV= confusion_matrix(z_test,predictsentNNCV)
classificationReportSentCV = classification_report(z_test,predictsentNNCV, zero_division=0)#warning occurs when model hasn't predicted a label, hence when making a report causes zero division warning
print(confusionMatrixSentCV)
print(classificationReportSentCV)

performance.write('\n\n classification report of grid search neural network for sentiment\n')
performance.write(classificationReportSentCV)
performance.write('\n\n confusion matrix of grid search neural network for sentiment\n')
npMatrixSentCV = np.array2string(confusionMatrixSentCV)
performance.write(npMatrixSentCV)
performance.write('\n\n-------------------------x-------------------------')