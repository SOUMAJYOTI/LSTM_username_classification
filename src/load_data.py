import pandas as pd
from collections import  *
import pickle

def loadData(file, vocab_map):
    data_df = pd.read_csv(file, header=None)
    data_df.columns = ["username"]

    # print(data_df[:10])
    # Form the vocabulary out of all characters
    count_chars = 0
    for idx, row in data_df.iterrows():
        uname = row['username'][1:]
        for i in range(len(uname)):
            if uname[i] not in vocab_map:
                vocab_map[uname[i]] = count_chars
                count_chars += 1

    return vocab_map

if __name__ == "__main__":
    vocab_map = defaultdict(str)

    file = "../data/pos_labels.txt"
    vocab_map = loadData(file, vocab_map)

    file = "../data/neg_labels.txt"
    vocab_map = loadData(file, vocab_map)

    pickle.dump(vocab_map, open("../data/vocab_map.pickle", "wb"))

