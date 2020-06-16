import pandas as pd
from peewee import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
"""
load database
"""
##db = MySQLDatabase("spider", host="127.0.0.1", port=3306, user="root", password="")
#df = pd.read_sql('SELECT * FROM review', con=db)
#df['rating'] = df['rating'].astype(float)

def split_dataframe(df, holdout_fraction=0.1):
  """Splits a DataFrame into training and test sets.
  Args:
    df: a dataframe.
    holdout_fraction: fraction of dataframe rows to use in the test set.
  Returns:
    train: dataframe for training
    test: dataframe for testing
  """
  test = df.sample(frac=holdout_fraction, replace=False)
  train = df[~df.index.isin(test.index)]
  return train, test


def MAE(R, u2w, w, u, b, c, mu):
    """
    predict rij = wi T uj + bi + cj + mu
    :return:
    """
    error = 0
    count = 0
    #print("MAE", len(u2w))
    for i in u2w.keys():
        for j in u2w[i]:
            pred = np.dot(w[i], u[j]) + b[i] + c[j] + mu
            error += np.abs(pred - R[i][j])
            count += 1

    return error / count



def ALS(Tr_u2w, Tr_w2u, Tv_u2w,  R, K, epochs, reg):
    """
    alternating least squares
    input:
    R: 2D array
    K : int, latent dimension
    epochs: int, number of iterations
    reg: real, regularization
    :return:
    list of real : loss at each epoch
    """
    Tr_losses = []
    Tv_losses = []
    N, M = rating.shape
    W = np.random.randn(N, K)
    b = np.zeros(N)
    U = np.random.randn(M, K)
    c = np.zeros(M)
    mu = np.mean(R[R != -1])
    for epoch in range(epochs):
        for i in Tr_u2w.keys():
            mat = np.eye(K) * reg
            vec = np.zeros(K)
            bi = 0
            count = 0
            for j in Tr_u2w[i]:
                r = rating[i][j]
                if r == -1:
                    continue
                count += 1
                mat += np.outer(U[j], U[j])
                vec += (r - b[i] - c[j] - mu) * U[j]
                bi += (r - W[i].dot(U[j]) - c[j] - mu)
            W[i] = np.linalg.solve(mat, vec)
            b[i] = bi / (count + reg)
        for j in Tr_w2u.keys():
            mat = np.eye(K) * reg
            vec = np.zeros(K)
            cj = 0
            count = 0
            for i in Tr_w2u[j]:
                r = rating[i, j]
                if r == -1:
                    continue
                count += 1
                mat += np.outer(W[i], W[i])
                vec += (r - b[i] - c[j] - mu) * W[i]
                cj += (r-W[i].dot(U[j]) - b[i] - mu)
            U[j] = np.linalg.solve(mat, vec)
            c[j] = cj / (count + reg)
        Tr_losses.append(MAE(R, Tr_u2w, W, U, b, c, mu))
        Tv_losses.append(MAE(R, Tv_u2w, W, U, b, c, mu))
        #print("epoch:{0:10d}, training loss={1:10.8f}, validation loss={2:10.8f}".format(epoch, Tr_losses[-1], Tv_losses[-1]))
    return Tr_losses, Tv_losses

def split(mat, ratio = 0.8):
    np.random.seed(0)
    valid = mat != -1
    idx = np.argwhere(valid)
    np.random.shuffle(idx)
    Tr_idx = idx[:int(len(idx) * ratio)]
    Tv_idx = idx[int(len(idx) * ratio):]
    print(len(Tr_idx), len(Tv_idx))
    Tr_u2w = defaultdict(list)
    Tr_w2u = defaultdict(list)
    Tv_u2w = defaultdict(list)
    Tv_w2u = defaultdict(list)
    for pair in Tr_idx:
        Tr_u2w[pair[0]].append(pair[1])
        Tr_w2u[pair[1]].append(pair[0])
    for pair in Tv_idx:
        Tv_u2w[pair[0]].append(pair[1])
        Tv_w2u[pair[1]].append(pair[0])
    return Tr_u2w, Tr_w2u, Tv_u2w, Tv_w2u

rating = np.load("training_mat.npy")



Tr_u2w, Tr_w2u, Tv_u2w, Tv_w2u = split(rating)
epochs = 40
best = 1000
for reg in [0.1, 1, 10]:
    for K in [2,5,10,20]:
    #for reg in [0.1, 1, 10, 100]:
        training_loss, validation_loss = ALS(Tr_u2w, Tr_w2u, Tv_u2w, rating, K, epochs, reg)
        best = np.min([best, validation_loss[-1]])
        print("K={0:5d}, reg={1:10.2f}, training loss={2:10.8f}, validation loss={3:10.8f}".format(K, reg, training_loss[-1], validation_loss[-1]))
print("best loss=", best)


