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
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
# fix random seed for reproducibility
from sklearn.metrics import precision_recall_fscore_support

np.random.seed(7)


def getFolds(Y, numFolds):
    # X = np.zeros(Y.shape[0])

    train_folds = []
    test_folds = []
    kf = KFold(n_splits=numFolds)
    cnt_folds = 0
    for train_index, test_index in kf.split(Y):
        train_folds.append(train_index)
        test_folds.append(test_index)

        cnt_folds += 1
        if cnt_folds >= numFolds:
            break

    return train_folds, test_folds


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

    return X, np.asarray(y)


def learnAndPredict(X, y, vocab_length):
    # truncate and pad input sequences
    max_uname_length = 15

    X = sequence.pad_sequences(X, maxlen=max_uname_length)

    # create the model
    embedding_vecor_length = 8
    model = Sequential()
    model.add(Embedding(vocab_length, embedding_vecor_length, input_length=max_uname_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    kf = ShuffleSplit(n_splits=10, test_size=.1, random_state=0)
    prec_pos = 0.
    rec_pos = 0.
    f1_pos = 0.
    prec_neg = 0.
    rec_neg=0.
    f1_neg=0.

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train,  epochs=50, batch_size=32, verbose=False)
        # Final evaluation of the model
        y_pred = model.predict_classes(X_test)

        y_pred_new = []
        for i in range(len(y_pred)):
            y_pred_new.append(y_pred[i][0])

        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_new)

        prec_pos += precision[1]
        rec_pos += recall[1]
        f1_pos += f1[1]

        prec_neg += precision[0]
        rec_neg += recall[0]
        f1_neg += f1[0]

        # scores = model.evaluate(X_test, y_test, verbose=0)
        # print("Accuracy: %.2f%%" % (scores[1]*100))

    print("Positive: ", prec_pos/10, rec_pos/10, f1_pos/10)
    print("negative:", prec_neg/10, rec_neg/10, f1_neg/10)


if __name__ == "__main__":
    pos_file = "../data/pos_labels.txt"
    neg_file = "../data/neg_labels.txt"

    data_pos_df = pd.read_csv(pos_file, header=None)
    data_pos_df.columns = ["username"]

    data_neg_df = pd.read_csv(neg_file, header=None)
    data_neg_df.columns = ["username"]

    vocab_map = pickle.load(open("../data/vocab_map.pickle", "rb"))
    X, y = prepareTrainTest(data_pos_df, data_neg_df, vocab_map)

    learnAndPredict(X, y, len(vocab_map))