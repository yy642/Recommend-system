import pandas as pd
import json
import numpy as np
"""
to-do
(1) memory-based -> base line done
(2) SVD 
(3) Deep learning?
(4) add reviews + deep learning?
"""
rating = np.load("training_mat.npy")

N_users, N_items = rating.shape
"""
compute the mean rating
"""
overall_mean = np.mean(rating[rating != -1])
mean_rating_user = np.zeros([N_users])
for i in range(N_users):
    rated = rating[i] != -1
    if np.sum(rated) == 0:
        mean_rating_user[i] = overall_mean
    else:
        mean_rating_user[i] = np.mean(rating[i][rated])

mean_rating_item = np.zeros([N_items])
for i in range(N_items):
    rated = rating[:, i] != -1
    if np.sum(rated) == 0:
        mean_rating_item[i] = overall_mean
    else:
        mean_rating_item[i] = np.mean(rating[:, i][rated])

def cos_similiarity(u, v):
    rated_u = u != -1
    rated_v = v != -1
    total = rated_u * rated_v
    if np.sum(total) == 0:
        return -1
    return np.dot(u[total],v[total])/np.sqrt(np.dot(u[rated_u], u[rated_u]) * np.dot(v[rated_v], v[rated_v]))



def user_user_similarity(rating):
    N_user = len(rating)
    similarity = np.zeros([N_user, N_user])
    for i in range(N_user):
        for j in range(i + 1, N_user):
            sim = cos_similiarity(rating[i], rating[j])
            similarity[i][j] = sim
            similarity[j][i] = sim
        similarity[i][i] = 1.0
    return similarity

def item_item_similarity(rating):
    N_item = len(rating[0])
    similarity = np.zeros([N_item, N_item])
    for i in range(N_item):
        for j in range(i + 1, N_item):
            sim = cos_similiarity(rating[:,i], rating[:,j])
            similarity[i][j] = sim
            similarity[j][i] = sim
        similarity[i][i] = 1.0
    return similarity

def predict_user_based(i, j, K, mean_rating, rating, sim_mat):
    """
    :param i: i-th user
    :param j: j-th product
    :param K: number of nearest neighbors considered
    :param mean_rating: average ratings by user
    :param rating: ratings by user
    :param sim_mat: user-user similarity matrix
    :return: predicted score
    """
    score = mean_rating[i]
    iprimes = np.argwhere(rating[:, j] != -1).flatten()
    if len(iprimes) == 0:
        return score
    sum = 0.
    sum2 = 0.
    for i_prime in iprimes:
        if np.sum((rating[i_prime] != -1) * ((rating[i] != -1))) >= K:
            sum += sim_mat[i, i_prime] * (rating[i_prime, j] - mean_rating[i_prime])
            sum2 += np.abs(sim_mat[i][i_prime])
    if sum2 == 0:
        return score
    score += sum / sum2
    return score

def predict_item_based(i, j, K, mean_rating, rating, sim_mat):
    """
    :param i: i-th user
    :param j: j-th product
    :param K: number of nearest neighbors considered
    :param mean_rating: average ratings by user
    :param rating: ratings by user
    :param sim_mat: user-user similarity matrix
    :return: predicted score
    """
    score = mean_rating[j]
    jprimes = np.argwhere(rating[i, :] != -1).flatten()
    if len(jprimes) == 0:
        return score
    sum = 0.
    sum2 = 0.
    for j_prime in jprimes:
        if np.sum((rating[:, j_prime] != -1) * (rating[:, j] != -1)) >= K:
            sum += sim_mat[j, j_prime] * (rating[i, j_prime] - mean_rating[j_prime])
            sum2 += np.abs(sim_mat[j][j_prime])
    if sum2 == 0:
        return score
    score += sum / sum2
    return score



sim_mat_user = user_user_similarity(rating)
sim_mat_item = item_item_similarity(rating)


def MSE(predict, K, rating, mean_rating, sim_mat, truth):
    errors=[]
    for i in range(len(truth)):
        for j in range(len(truth[0])):
            if truth[i][j] != -1:
                errors.append(np.abs((predict(i, j, K, mean_rating, rating, sim_mat) - truth[i, j])))
    return [np.mean(errors), np.std(errors)]
testing = np.load("testing_mat.npy")
for K in [1,2,3,4,5]:
    print("K=", K)
    #print(MSE(predict_user_based, K, rating, mean_rating_user, sim_mat_user, rating))
    #print(MSE(predict_item_based, K, rating, mean_rating_item, sim_mat_item, rating))
    print(MSE(predict_user_based, K, rating, mean_rating_user, sim_mat_user, testing))
    print(MSE(predict_item_based, K, rating, mean_rating_item, sim_mat_item, testing))