import torch
import numpy as np
import matplotlib.pyplot as plt


def cal_accuracy(prediction, labels):
    prediction = prediction.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    hypothesis = [np.argmax(predict) for predict in prediction]
    same = (hypothesis == labels)
    return np.sum(same) / same.shape[0]


def cal_number(prediction, labels):
    prediction = prediction.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    hypothesis = [np.argmax(predict) for predict in prediction]
    same = (hypothesis == labels)
    return np.sum(same)


def write_list(lists, path):
    with open(path, 'w') as file:
        file.write(' '.join(list(map(str, lists))))


def read_list(path):
    with open(path, 'r') as file:
        lists = list(map(float, file.read().split()))
    return lists




