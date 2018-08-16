import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
np.random.seed(7)


def convertWordToNumericList(word, vocab_map):
    '''

    :param word:
    :return:
    '''

    numList = []
    for i in range(1, len(word)):
        numList.append( vocab_map[word[i]])

    return numList


def prepareTrainTest(pos_data, neg_data, vocab_map):
    '''

    :param pos_data:
    :param neg_data:
    :param vocab_map:
    :return:
    '''

    X = []
    y = []
    for idx, row in pos_data.iterrows():
        uname = row['username']
        nameNumList = convertWordToNumericList(uname, vocab_map)
        X.append(nameNumList)
        y.append(1.)


    for idx, row in neg_data.iterrows():
        uname = row['username']
        nameNumList = convertWordToNumericList(uname, vocab_map)
        X.append(nameNumList)
        y.append(0.)

    print(X[:10])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print(X_train[:10])
    return X_train, X_test, y_train, y_test


def learnAndPredict(X_train, X_test, y_train, y_test, vocab_length):
    # truncate and pad input sequences
    max_uname_length = 10

    X_train = sequence.pad_sequences(X_train, maxlen=max_uname_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_uname_length)

    # create the model
    embedding_vecor_length = 8
    model = Sequential()
    model.add(Embedding(vocab_length, embedding_vecor_length, input_length=max_uname_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    pos_file = "../data/pos_labels.txt"
    neg_file = "../data/neg_labels.txt"

    data_pos_df = pd.read_csv(pos_file, header=None)
    data_pos_df.columns = ["username"]

    data_neg_df = pd.read_csv(neg_file, header=None)
    data_neg_df.columns = ["username"]

    vocab_map = pickle.load(open("../data/vocab_map.pickle", "rb"))
    X_train, X_test, y_train, y_test = prepareTrainTest(data_pos_df, data_neg_df, vocab_map)

    learnAndPredict(X_train, X_test, y_train, y_test, len(vocab_map))