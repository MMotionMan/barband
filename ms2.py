import numpy as np
import faiss
import sympy as sp
import scipy
from navec import Navec
import navec
import pandas as pd
import math

'''
Read an index
'''
x = sp.symbols('x')


def select_last_k(user_list, k):
    return user_list[-k:]


def create_weighted_dist(D, numerator=1, degree=1, function=None):
    return numerator / (D + 1e-4) ** degree


def create_time_punishment_vector(k, function=x, numerator=1):
    step = 0.5
    start_pun = 2.0
    steps = np.arange(start_pun, (start_pun + step * k) - (1e-6), step)
    v_weigted = [float(numerator / function.subs(x, i)) for i in steps]

    return np.array(v_weigted)


def combine_weights_and_ids(coefficients, I, probs):
    d = {}
    array_prob = []
    item_ids = []
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            d[I[i, j]] = [coefficients[i, j], probs[i, j]]
            array_prob.append(probs[i, j])
            item_ids.append(I[i, j])
    return d, array_prob, item_ids


def softmax(vec, temperature):
    """
    turn vec into normalized probability
    """
    sum_exp = sum(math.exp(x / temperature) for x in vec)
    return [math.exp(x / temperature) / sum_exp for x in vec]


# input: list of item_ids visited by user_i. List of int
def concierge_service(user_list: list):
    flat_index = faiss.read_index("./online_flat.index")
    query = faiss.read_index("./query.index")
    user_list = np.array(user_list)
    last_k_visited = 0
    # Select the last n visited events
    if len(user_list) > 5:
        last_k_visited = 5
    else:
        last_k_visited = len(user_list)
    user_relevant_list = select_last_k(user_list, last_k_visited)[::-1]
    distances = []
    closest_indices = []
    for i in user_relevant_list:
        query_vec = query.reconstruct(int(i))
        D, I = flat_index.search(np.array([query_vec]), 6)
        if len(distances) == 0:
            distances = D
            closest_indices = I
        else:
            distances = np.concatenate((distances, D), axis=0)
            closest_indices = np.concatenate((closest_indices, I), axis=0)
        for rem_index in I[0]:
            flat_index.remove_ids(np.array([int(rem_index)]).astype('int64'))

    D = distances
    I = closest_indices
    D_div_1 = create_weighted_dist(D, degree=1, numerator=100)
    x = sp.symbols('x')

    # f = x ** 2
    # f = x #Okey
    # f = x**1.1 #Okey
    # f = x**100
    # f = sp.sqrt(x)
    f = sp.sqrt(sp.sqrt(x))
    # f = sp.log(x)

    # Create punishment coefficient for each recommended events because last visited events affect less than new visited events
    time_vector = create_time_punishment_vector(k=last_k_visited, function=f, numerator=1)
    D_div_1 = np.transpose(D_div_1)
    coefficients = np.transpose(time_vector * D_div_1)
    median_weight = np.median(coefficients)
    n_rows = coefficients.shape[0]
    n_cols = coefficients.shape[1]

    low_median_coefficient = 0.3
    up_median_coefficient = 2
    for i in range(n_rows):
        for j in range(n_cols):
            if coefficients[i, j] < median_weight:
                coefficients[i, j] *= low_median_coefficient
            else:
                coefficients[i, j] *= up_median_coefficient

    coefficients_vector = np.reshape(coefficients, (n_rows * n_cols,))
    soft_max_vec = softmax(coefficients_vector, 1.8)
    probs = np.reshape(soft_max_vec, (n_rows, n_cols))
    dict_w1, array_prob, item_ids = combine_weights_and_ids(coefficients, I, probs)
    ranks = np.random.choice(item_ids, size=len(item_ids), replace=False, p=array_prob)
    return list(ranks)

