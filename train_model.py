#!/usr/bin/python
# -*- coding: utf-8 -*-

from lstm import *
from read_dataset import *
from Evaluate import *
from util import *
import os
import datetime

_HIDDEN_DIM_ = int(os.environ.get('HIDDEN_DIM', '13'))
_LEARNING_RATE_ = float(os.environ.get('LEARNING_RATE', 0.01))
_NEPOCH_ = int(os.environ.get('NEPOCH', '200'))
_MODEL_FILE_ = os.environ.get('MODEL_FILE')
_TRAINED_MODEL_ = os.environ.get("TRAINED_MODEL")


# predict with tune data
def predict_tune_file(model, tune_x_data):
    print "start testing..."
    avg_mean_average_precision = 0.0
    avg_recall_1 = 0.0
    avg_recall_5 = 0.0
    avg_recall_10 = 0.0
    for x_key in tune_x_data:
        u = x_key
        x_list = tune_x_data[x_key]
        max_length = 1000
        mean_average_precision = 0.0
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
                    probability = model.forward_propagation(x, u)
                    mean_average_precision += MAP(probability, y)
                    r_1 += recall_1(predict, y)
                    r_5 += recall_5(predict, y)
                    r_10 += recall_10(predict, y)
                else:
                    x = x_list[i * max_length:-1]
                    y = x_list[i * max_length + 1:]
                    if len(x) == 0:
                        continue
                    predict = model.predict(x, u)
                    probability = model.forward_propagation(x, u)
                    mean_average_precision += MAP(probability, y)
                    r_1 += recall_1(predict, y)
                    r_5 += recall_5(predict, y)
                    r_10 += recall_10(predict, y)
        else:
            x = x_list[:-1]
            y = x_list[1:]
            if len(x) == 0:
                continue
            predict = model.predict(x, u)
            # print predict
            # print y
            probability = model.forward_propagation(x, u)
            mean_average_precision += MAP(probability, y)
            r_1 += recall_1(predict, y)
            r_5 += recall_5(predict, y)
            r_10 += recall_10(predict, y)
        # MRR for each poi of the user
        mean_average_precision /= len(x_list)
        r_1 /= len(x_list)
        r_5 /= len(x_list)
        r_10 /= len(x_list)

        avg_mean_average_precision += mean_average_precision
        avg_recall_1 += r_1
        avg_recall_5 += r_5
        avg_recall_10 += r_10
    # average MRR for users
    avg_mean_average_precision /= len(tune_x_data)
    avg_recall_1 /= len(tune_x_data)
    avg_recall_5 /= len(tune_x_data)
    avg_recall_10 /= len(tune_x_data)

    print "the average MAP score for test text is: ", avg_mean_average_precision
    print "the average recall@1 is: ", avg_recall_1
    print "the average recall@5 is: ", avg_recall_5
    print "the average recall@10 is: ", avg_recall_10


def train_with_sgd(model, x_train, learning_rate=0.005, nepoch=1):
    print "start training..."

    last_loss = 500000.0
    for epoch in range(nepoch):
        print "epoch ", epoch
        loss = 0.0
        for x_key in x_train:
            u = x_key
            data_list = x_train[x_key]
            # get the input poi -- current time step
            x = data_list[:-1]
            # get the output correct poi -- next time step
            y = data_list[1:]
            # print(x)
            # print(index)
            model.sdg_step(x, y, u, learning_rate)
            # print model.forward_propagation(x, u)
            # print model.cal_loss_function(x, y, u)
            # print "finished training user %s" % u, datetime.datetime.now()
            loss += model.cal_loss_function(x, y, u)
        print "loss: ", loss
        print datetime.datetime.now()
        predict_tune_file(model, tune_x_data)

        # adjust the learning rate according to the loss reduced
        if last_loss - loss < 0.1 * loss:
            learning_rate *= 0.8
        print "epoch ", epoch, " completed..."

    # time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # save_model_parameters_theano("./lstm-theano-%d-%s" % (model.hidden_dim, time), model)

if not _MODEL_FILE_:
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    _MODEL_FILE_ = "LSTM-%s-%s-%s.dat" % (_HIDDEN_DIM_, _LEARNING_RATE_, _NEPOCH_)

print "Reading training file..."
# read train data
train_file_path = '../GTD/train.txt'
x_data = read_gtd_data(train_file_path)
num_user = read_gtd_users()
num_poi = read_gtd_pois()

# read tunedata
tune_file_path = '../GTD/test.txt'
tune_x_data = read_gtd_data(tune_file_path)

if not _TRAINED_MODEL_:
    # train model with training data
    print "building model..."
    model = LSTMtheano(hidden_dim=_HIDDEN_DIM_, num_user=num_user, num_poi=num_poi)
    print datetime.datetime.now()
else:
    # trained with existed model
    model = load_model_parameters_lstm(path=_TRAINED_MODEL_)

train_with_sgd(model, x_train=x_data, learning_rate=_LEARNING_RATE_, nepoch=_NEPOCH_)

# after training, we save the model parameters
save_model_parameters_lstm(model, _MODEL_FILE_)

# predict_tune_file(model, tune_x_data)
