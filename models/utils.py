import torch
from torch.nn import functional as F
import argparse
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def dcg(scores):
    """
    compute the DCG value based on the given score
    :param scores: a score list of documents
    :return v: DCG value
    """
    v = 0
    for i in range(len(scores)):
        v += (np.power(2, scores[i]) - 1) / np.log2(i + 2)  # i+2 is because i starts from 0
    return v


def idcg(scores):
    """
    compute the IDCG value (best dcg value) based on the given score
    :param scores: a score list of documents
    :return:  IDCG value
    """
    best_scores = sorted(scores)[::-1]
    return dcg(best_scores)


def ndcg(scores):
    """
    compute the NDCG value based on the given score
    :param scores: a score list of documents
    :return:  NDCG value
    """
    return dcg(scores) / idcg(scores)


def single_dcg(scores, i, j):
    """
    compute the single dcg that i-th element located j-th position
    :param scores:
    :param i:
    :param j:
    :return:
    """
    return (np.power(2, scores[i]) - 1) / np.log2(j + 2)


def delta_ndcg(scores, p, q):
    """
    swap the i-th and j-th doucment, compute the absolute value of NDCG delta
    :param scores: a score list of documents
    :param p, q: the swap positions of documents
    :return: the absolute value of NDCG delta
    """
    s2 = scores.copy()  # new score list
    s2[p], s2[q] = s2[q], s2[p]  # swap
    return abs(ndcg(s2) - ndcg(scores))


def ndcg_k(scores, k):
    scores_k = scores[:k]
    dcg_k = dcg(scores_k)
    idcg_k = dcg(sorted(scores)[::-1][:k])
    if idcg_k == 0:
        return np.nan
    return dcg_k / idcg_k


def group_by(data, qid_index):
    """

    :param data: input_data
    :param qid_index: the column num where qid locates in input data
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    for record in data:
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    return qid_doc_map


def get_pairs(scores):
    """
    compute the ordered pairs whose firth doc has a higher value than second one.
    :param scores: given score list of documents for a particular query
    :return: ordered pairs.  List of tuple, like [(1,2), (2,3), (1,3)]
    """
    pairs = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
    return pairs


def split_pairs(order_pairs, true_scores):
    """
    split the pairs into two list, named relevant_doc and irrelevant_doc.
    relevant_doc[i] is prior to irrelevant_doc[i]

    :param order_pairs: ordered pairs of all queries
    :param ture_scores: scores of docs for each query
    :return: relevant_doc and irrelevant_doc
    """
    relevant_doc = []
    irrelevant_doc = []
    doc_idx_base = 0
    query_num = len(order_pairs)
    for i in range(query_num):
        pair_num = len(order_pairs[i])
        docs_num = len(true_scores[i])
        for j in range(pair_num):
            d1, d2 = order_pairs[i][j]
            d1 += doc_idx_base
            d2 += doc_idx_base
            relevant_doc.append(d1)
            irrelevant_doc.append(d2)
        doc_idx_base += docs_num
    return relevant_doc, irrelevant_doc


def compute_lambda(true_scores, temp_scores, order_pairs, qid):
    """

    :param true_scores: the score list of the documents for the qid query
    :param temp_scores: the predict score list of the these documents
    :param order_pairs: the partial oder pairs where first document has higher score than the second one
    :param qid: specific query id
    :return:
        lambdas: changed lambda value for these documents
        w: w value
        qid: query id
    """
    doc_num = len(true_scores)
    lambdas = np.zeros(doc_num)
    w = np.zeros(doc_num)
    IDCG = idcg(true_scores)
    single_dcgs = {}
    for i, j in order_pairs:
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

    for i, j in order_pairs:
        delta = abs(single_dcgs[(i, j)] + single_dcgs[(j, i)] - single_dcgs[(i, i)] - single_dcgs[(j, j)]) / IDCG
        rho = 1 / (1 + np.exp(temp_scores[i] - temp_scores[j]))
        lambdas[i] += rho * delta
        lambdas[j] -= rho * delta

        rho_complement = 1.0 - rho
        w[i] += rho * rho_complement * delta
        w[i] -= rho * rho_complement * delta

    return lambdas, w, qid


def load_data(file_path='/Users/hou/OneDrive/KDD2019/data/L2R/sample.txt'):
    with open(file_path, 'r') as f:
        data = []
        for line in f.readlines():
            new_arr = []
            line_split = line.split(' ')
            score = float(line_split[0])
            qid = int(line_split[1].split(':')[1])
            new_arr.append(score)
            new_arr.append(qid)
            for ele in line_split[2:]:
                new_arr.append(float(ele.split(':')[1]))
            data.append(new_arr)
    data_np = np.array(data)
    return data_np


def decay_learning_rate(optimizer, epoch, decay_rate):
    if (epoch + 1) % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
