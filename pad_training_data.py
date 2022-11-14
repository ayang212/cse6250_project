import pandas as pd
import numpy as np
import os

def load_data(path):
    def get_sample_len(row):
        len = row.shape[0]
        return len
    
    def pad_training_data(df):
        # data_np => padded training data
        data_np = np.zeros((202,1303,100))
        for i, data in enumerate(df['data']):
            n = data.shape[0]
            data_np[i,:n,:] = data
        
        return data_np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # df: dataframe containing 3 columns (id, data, labels)
    df = pd.read_json(path)['training_data'].apply(pd.Series)
    df['data'] = df['data'].apply(np.array)
    df['lens'] = df['data'].apply(get_sample_len)

    #### max len is 2978 for training data non-unique tokens
    #### max len is 1303 for training data unique tokens
    # unique tokens -> apply set() to list of tokens before obtaining word vectors
    
    padded_training_data = pad_training_data(df)
    
    


path = 'training_data_w_sets.json'
# path = 'training_data.json'
load_data(path)
