import numpy as np
import pandas as pd


def get_dict(file_name):
    my_file = pd.read_csv(file_name, delimiter=' ')
    file = {}
    for i in range(len(my_file)):
        en = my_file.iloc[i][0]
        fr = my_file.iloc[i][1]
        file[en] = fr

    return file


def cosine_similarity(A, B):
    axb = np.dot(A, B)
    a_norm = np.linalg.norm(A)
    b_norm = np.linalg.norm(B)
    cos_b = axb/(a_norm * b_norm)
    return cos_b
