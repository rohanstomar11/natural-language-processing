import numpy as np

def get_vecs(embeddings, words):
    m = len(words)
    X = np.zeros((1,300))
    for word in words:
        x = word
        embedd = embeddings[x]
        X = np.row_stack((X, embedd))
    X=X[1:,:]    
    return X