import pandas as pd 

def read_txt(filename, separator=" ", header=None):
    return pd.read_csv(filename, sep=separator, header=header).to_numpy()