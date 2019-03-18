import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def onehotencode(data):
    final_res = []
    for singledata in data:
        res = []
        for char in singledata:
            if char in 'AUGC':
                res.append(char)
        final_res.append(get_onehotencoding(res))
    return final_res




def get_onehotencoding(data):
    values = np.array(data)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

def load_data(file_path):
    data = pd.read_csv(file_path)
    data_seq = data['Seq'].tolist()
    data_seq = np.array(onehotencode(data_seq))
    label = data['Label'].tolist()
    return data_seq, label

if __name__ == '__main__':
    data, label = load_data()
    print(label)
    print(data)