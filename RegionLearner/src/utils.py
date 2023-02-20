import pandas as pd 

def read_txt(filename, separator=" ", header=None):
    """Utility function to read a txt file.

    Args:
        filename (str): the file name.
        separator (str, optional): the separator for the values inside the file. Defaults to " ".
        header (list(str), optional): the names of the columns. Defaults to None.

    Returns:
        ndarray: the values read from the txt file.
    """    
    return pd.read_csv(filename, sep=separator, header=header).to_numpy()