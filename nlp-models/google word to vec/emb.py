import pandas as pd
import numpy as np
import gensim
import nltk
from gensim.models import Word2Vec, KeyedVectors
from nltk.internals import Counter
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neural_network
import gensim.downloader as download
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.probability import FreqDist
import sys

file = open(sys.path[0] + '/../../Q1/data.csv', encoding="utf-8")
#____________________________________Q3.1
# download the model and save to computer
#model_google_news = download.load("word2vec-google-news-300", return_path= True)
# load the model
google_300_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
#print(google_300_model['hello'].shape) #test if model works

hyperparameters = {
    'activation': ['relu','tanh'],  # , ,'identity', 'logistic''relu',
    'hidden_layer_sizes': [[30, 50],[10,10,10]],  #
    'solver': ['adam','sgd']  # 'sgd
}

performance = open("performance for google news 300 word to vec.txt", "w")
data = pd.read_csv(file, names=['post', 'emotion', 'sentiment'], header=0)

df_x = data['post'].dropna()  # first column of posts without NaN
df_y = data['emotion'].dropna()  # second column of emotion without NaN
df_z = data['sentiment'].dropna()  # third column of sentiment without NaN
le = LabelEncoder()
Y = le.fit_transform(df_y)
Z = le.fit_transform(df_z)
# split post into 80 training and 20 test, needs to be done without randomizing index
x_train, x_test, y_train, y_test = train_test_split(df_x, Y, test_size=0.2, random_state=50)

# tokenize the reddit post
post = df_x.apply(nltk.tokenize.word_tokenize)

# remove punctuations by the following
regex = RegexpTokenizer(r'\w+')  # source https://www.nltk.org/api/nltk.tokenize.regexp.html
post_noPunc = df_x.apply(regex.tokenize)  # has the words from reddit post, without punctuations

#____________________________________Q3.2
def uniqueTokens(post):  # takes in tokenized data
    # the following code produces the total unique tokens using nltk library, takes a lot of time
    sum = []
    for i in range(len(post)):
        sum = sum + post[i]
    # sum has all the tokens in one array
    count = FreqDist(sum)
    return count  # has the number of samples (total unique tokens) and outcomes (total tokens)


def tokensCount(post):  # takes in tokenized data
    # following function counts all the tokens in the reddit posts tokenized using nltk library (does contain
    # repeated values)
    sumPost = 0
    for i in range(len(post)):
        sumPost = sumPost + len(post[i])
    return sumPost


def tokensCountNoPunc(post_noPunc):  # takes in tokenized data
    # finding the length of tokens/words in training sets with no punctuations (does contain repeated values)
    sumNoPunc = 0
    for i in range(len(post_noPunc)):
        sumNoPunc = sumNoPunc + len(post_noPunc[i])
    return sumNoPunc


#print('total unique tokens in the reddit posts: ',uniqueTokens(post))
print('total tokens for reddit post in training set is: ', tokensCount(post))
print('total tokens for reddit post without punctuations in training set is: ', tokensCountNoPunc(post_noPunc))



#____________________________________Q3.3
def postEmb(post, x):  # takes in post number and post
    # function to create embedding of a post as average of the embeddings of its words(skipped if not present)
    sumEmbedd = 0
    totalCount = 0
    avg = 0
    for i in range(len(post[x])):
        try:
            sumEmbedd = sumEmbedd + google_300_model[post[x][i]]
            totalCount = totalCount + 1
        except KeyError:
            pass  # print(post[x][i], 'not in model')

    if totalCount == 0:# when no words have embeddings
        avg = np.array([0 for i in range(300)])
    else:
        avg = sumEmbedd / totalCount
    return avg


print('embedding of a reddit post 10 as the average of embeddings of its words', postEmb(post, 10))

#____________________________________Q3.4
# overall hit rates
# note nltk library's freq dist was not used to get the individual tokens, due to computer constrains

def hitRateunique(data):  # takes the entire data set
    # this function produces the hit rate for all the unique words in token
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    vocab_reddit = vectorizer.get_feature_names_out()  # has all the unique words of the vocab from reddit post
    success = 0
    fail = 0

    # go through reddit vocab and find words that have vector vs don't
    for i in range(len(vocab_reddit)):
        try:
            google_300_model[vocab_reddit[i]]
            success = success + 1
        except KeyError:
            fail = fail + 1
    hit_rate_value = ((success) / len(vocab_reddit)) * 100
    return hit_rate_value


def hitRate(post):  # takes in tokenized data
    # this function produces the hit rate for all the words/token from nltk tokenizer, and will contain repeat words.
    success = 0
    fail = 0
    totalTokens = tokensCount(post)  # total token in the tokenized data of reddit post

    # go through reddit vocab and find words that have vector vs don't
    for i in range(len(post)):
        for j in range(len(post[i])):
            try:
                google_300_model[post[j]]
                success = success + 1
            except KeyError:
                fail = fail + 1
    hit_rate_value = ((success) / totalTokens) * 100
    return hit_rate_value


print('hit rate for just the unique words of the vocab(using countvectorizer)',hitRateunique(df_x))
print('hit rate for all the words of the vocab(using nltk)',hitRate(post))


#____________________________________Q3.5/3.6/3.7
# neural network
def createEmbVector(post):  # takes in tokenized data
    embVector = np.empty(shape=(len(post), 300))
    embVector.fill(0)
    for i in range(len(post)):
        embVector[i] = postEmb(post, i)

    return embVector


X = createEmbVector(post)   # create the vector with embedding of each post, as the average of the embedding
# of all the words the post contains


def base_neural(X_train, X_text, y_train, y_test):
    baseNeural = neural_network.MLPClassifier(max_iter=2)
    modelBase = baseNeural.fit(X_train, y_train)
    predictBase = modelBase.predict(X_text)
    score = accuracy_score(y_test, predictBase)
    classificationReport = classification_report(y_test, predictBase, zero_division=0)
    performance.write('classification report of base neural network\n\n')
    performance.write(str(classificationReport))
    return score


def grid_neural(x_train, X_text, y_train, y_test):
    gridNeural = neural_network.MLPClassifier(max_iter=2)
    grid = GridSearchCV(estimator=gridNeural, param_grid=hyperparameters)
    modelGrid = grid.fit(x_train, y_train)
    predictGrid = modelGrid.predict(X_text)
    scoreGrid = accuracy_score(y_test, predictGrid)
    classificationReport = classification_report(y_test, predictGrid, zero_division=0)
    performance.write('\n\nclassification report of grid search neural network\n\n')
    performance.write(str(classificationReport))
    performance.write('\nbest parameter for grid search:')
    performance.write(str(grid.best_params_))
    performance.write('\n\n')
    return scoreGrid


# split dataset for emotion
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# emotion
performance.write('\nQ3 emotion metrics\n')
print('base neural network accuracy score for emotion is :', base_neural(X_train, X_test, Y_train, Y_test))
print('grid search neural network accuracy score for emotion is :', grid_neural(X_train, X_test, Y_train, Y_test))

performance.write('\n----------------------------x---------------------------\n')


# split dataset for sentiment
x2_train, x2_test, z_train, z_test = train_test_split(X, Z, test_size=0.2, random_state=50)
# sentiment
performance.write('\nQ3 sentiment metrics\n')
print('base neural network accuracy score for sentiment is :', base_neural(x2_train, x2_test, z_train, z_test))
print('grid search neural network accuracy score for sentiment is :', grid_neural(x2_train, x2_test, z_train, z_test))
performance.write('\n----------------------------x---------------------------\n')

performance.close()