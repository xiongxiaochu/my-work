#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import linalg as la
import numpy as np


def get_topK(out_t, top_K):
    poi_list = []
    # get the top_K pois in terms of probability
    a = out_t
    for i in range(0, top_K):
        index = np.ndarray.argmax(a)
        poi_list.append(int(index))
        np.put(a, index, -1)
    return poi_list


# calculate the MAP score for a user
def MAP(probability, y):
    prob = np.argsort(-probability, axis=1)
    rank = 0.0
    for i in np.arange(len(y)):
        for j in np.arange(len(prob[i])):
            if int(prob[i][j]) == y[i]:
                rank += 1.0 / (j + 1)
                break
    return rank


# calculate the recall score for a user
def recall_1(predict, y):
    recall = 0
    for i in np.arange(len(y)):
        if int(predict[i][0]) == y[i]:
            recall += 1
            break
    return recall


def recall_5(predict, y):
    recall = 0
    for i in np.arange(len(y)):
        for j in range(5):
            if int(predict[i][j]) == y[i]:
                recall += 1
                break
    return recall


def recall_10(predict, y):
    recall = 0
    for i in np.arange(len(y)):
        for j in range(10):
            if int(predict[i][j]) == y[i]:
                recall += 1
                break
    return recall


def recall_100(predict, y):
    recall = 0
    for i in np.arange(len(y)):
        for j in range(100):
            if int(predict[i][j]) == y[i]:
                recall += 1
                break
    return recall


def recall_500(predict, y):
    recall = 0
    for i in np.arange(len(y)):
        for j in range(500):
            if int(predict[i][j]) == y[i]:
                recall += 1
                break
    return recall


def precision_1(predict, y):
    precision = 0
    for i in np.arange(len(y)):
        if int(predict[i][0]) == y[i]:
            precision += 1
            break
    return precision


def precision_5(predict, y):
    precision = 0.0
    for i in np.arange(len(y)):
        for j in range(5):
            if int(predict[i][j] == y[i]):
                precision += 1.0 / (j + 1)
                break
    return precision


def precision_10(predict, y):
    precision = 0.0
    for i in np.arange(len(y)):
        for j in range(10):
            if int(predict[i][j] == y[i]):
                precision += 1.0 / (j + 1)
                break
    return precision
