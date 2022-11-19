import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split

def load_data(path, data_type):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # df: dataframe containing 3 columns (id, data, labels)
    df = pd.read_json(path)[data_type].apply(pd.Series)
    df['data'] = df['data'].apply(np.array)

    # get lengths of each np array and sort for padding
    df['lengths'] = df['data'].apply(lambda x: x.shape[0])
    df = df.sort_values(by=['lengths'], ascending=False)

    x = pad_training_data(df)
    y = df['labels']
    lengths = df['lengths']
    
    return x, y, lengths


def pad_training_data(df):
    # data_np => padded training data
    max_length = df['lengths'].max()
    data_size = df.shape[0]
    np_dataset = np.zeros((data_size,max_length,100))
    for i, np_doc in enumerate(df['data']):
        n = np_doc.shape[0]
        np_dataset[i,:n,:] = np_doc
    
    return np_dataset

def load_tensors(x,y,lengths):

    y_list = []
    for idx, label in enumerate(y):
        y_i = [v for k,v in label.items()]
        y_list.append(y_i)
    y_np = np.array(y_list)
    data = torch.from_numpy(x.astype('float32'))
    target = torch.from_numpy(y_np.astype('long')).long()
    dataset = TensorDataset(data, target, torch.from_numpy(np.array(lengths)))
    return dataset, list(label.keys())

def main():
    path = 'training_data.json'
    x,y,lengths = load_data(path)
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(x,y,lengths, test_size=0.2, random_state=1)

    train_tensor = load_tensors(X_train,y_train,len_train)
    validation_tensor = load_tensors(X_test,y_test,len_test)

if __name__ == "__main__":
    main()
