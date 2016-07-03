#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import os
from RNNtheano import *
from util import *
from read_dataset import *
import numpy as np

_HIDDEN_DIM_ = int(os.environ.get('HIDDEN_DIM', '30'))
_LEARNING_RATE_ = float(os.environ.get('LEARNING_RATE', 0.04))
_NEPOCH_ = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE_ = os.environ.get('MODEL_FILE')
_TRAINED_MODEL_ = os.environ.get("TRAINED_MODEL")


def evaluate(model, tune_x_data):
    # predict with test data
    avg_mean_reciprocal_rank = 0.0
    avg_recall_1 = 0.0
    avg_recall_5 = 0.0
    avg_recall_10 = 0.0
    for x_key in tune_x_data:
        u = x_key
        x_list = tune_x_data[x_key]
        max_length = 300
        mean_reciprocal_rank = 0.0
        r_1 = 0.0
        r_5 = 0.0
        r_10 = 0.0
        # if the list is too long, we split it into several folds to avoid memory lacking
        if len(x_list) > max_length:
            fold = len(x_list) / max_length
            for i in range(fold + 1):
                if i < fold:
                    x = x_list[i * max_length:(i+1) * max_length - 1]
                    y = x_list[i * max_length + 1:(i+1) * max_length]
                    predict = model.predict(x, u)
                    prob = model.forward_propagation(x, u)
                    r_1 += recall_1(predict, y)
                    r_5 += recall_5(predict, y)
                    r_10 += recall_10(predict, y)
                    mean_reciprocal_rank += MAP(prob, y)
                else:
                    x = x_list[i * max_length:-1]
                    y = x_list[i * max_length + 1:]
                    if len(x) == 0:
                        continue
                    predict = model.predict(x, u)
                    prob = model.forward_propagation(x, u)
                    r_1 += recall_1(predict, y)
                    r_5 += recall_5(predict, y)
                    r_10 += recall_10(predict, y)
                    mean_reciprocal_rank += MAP(prob, y)
        else:
            x = x_list[:-1]
            y = x_list[1:]
            if len(x) == 0:
                continue
            predict = model.predict(x, u)
            prob = model.forward_propagation(x, u)
            r_1 += recall_1(predict, y)
            r_5 += recall_5(predict, y)
            r_10 += recall_10(predict, y)
            mean_reciprocal_rank = MAP(prob, y)
        # MRR for each poi of the user
        mean_reciprocal_rank /= len(x_list)
        r_1 /= len(x_list)
        r_5 /= len(x_list)
        r_10 /= len(x_list)

        avg_mean_reciprocal_rank += mean_reciprocal_rank
        avg_recall_1 += r_1
        avg_recall_5 += r_5
        avg_recall_10 += r_10
    # average MRR for users
    avg_mean_reciprocal_rank /= len(tune_x_data)
    avg_recall_1 /= len(tune_x_data)
    avg_recall_5 /= len(tune_x_data)
    avg_recall_10 /= len(tune_x_data)

    print "the average MAP score for tune text is: ", avg_mean_reciprocal_rank
    print "the average recall@1 is: ", avg_recall_1
    print "the average recall@5 is: ", avg_recall_5
    print "the average recall@10 is: ", avg_recall_10


def train_with_sgd(model, x_train, learning_rate=0.05, nepoch=1):
    print datetime.datetime.now()
    print "start training..."
    batch_size = 40
    for epoch in range(nepoch):
        num = 1
        loss = 0.0
        print "epoch ", epoch
        for x_key in x_train:
            u = x_key
            x_list = x_train[x_key]
            i = 0
            x = []
            y = []
            for x_i in x_list:
                y_i = []
                # negative sampling
                for j in range(10):
                    neg = np.random.randint(0, num_poi)
                    while 1:
                        if neg != x_i:
                            y_i.append(neg)
                            break
                        else:
                            neg = np.random.randint(0, num_poi)
                x.append(x_i)
                y.append(y_i)
                i += 1
                if i % batch_size == 0:
                    model.sdg_step(x, y, u, learning_rate, batch_size)
                    # print datetime.datetime.now()
                    # print model.forward(x, y, u)
                    loss += model.cal_loss_function(x, y, u)
                    x = []
                    y = []
                elif i == len(x_list):
                    if len(x) == 0:
                        break
                    model.sdg_step(x, y, u, learning_rate, batch_size)
                    # print datetime.datetime.now()
                    # print model.forward(x, y, u)
                    loss += model.cal_loss_function(x, y, u)
            print "finished training user %d" % num, datetime.datetime.now()
            # loss /= len(x_list)
            num += 1
        # loss /= len(x_train)

        print "loss: ", loss
        print datetime.datetime.now()
        evaluate(model, tune_x_data)
        print "epoch ", epoch, " completed..."

    # time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # save_model_parameters_theano("./lstm-theano-%d-%s" % (model.hidden_dim, time), model)

if not _MODEL_FILE_:
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    _MODEL_FILE_ = "RNN-%s-%s.dat" % (ts, _HIDDEN_DIM_)

print "Reading training file..."
print datetime.datetime.now()
# read train data
train_file_path = '../gowalla/train.txt'
test_file_path = '../gowalla/test.txt'
x_data = read_gowalla_data(train_file_path)
tune_x_data = read_gowalla_data(test_file_path)
num_user = read_gowalla_users()
num_poi = read_gowalla_pois()

if not _TRAINED_MODEL_:
    # train model with training data
    print datetime.datetime.now()
    print "building model..."
    model = RNNtheano(hidden_dim=_HIDDEN_DIM_, num_user=num_user, num_poi=num_poi)
else:
    # trained with existed model
    model = load_model_parameters_rnn1(path=_TRAINED_MODEL_)

train_with_sgd(model, x_train=x_data, learning_rate=_LEARNING_RATE_, nepoch=_NEPOCH_)

# after training, we save the model parameters
# save_model_parameters_rnn(model, _MODEL_FILE_)


