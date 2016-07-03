import numpy as np
import sys
from rnn import *
from read_dataset import *
from LSTMtheano import *
from RNNtheano import *
from LSTM_neg import *


def save_model_parameters_lstm(model, outfile):
    np.savez(outfile,
             Wi=model.Wi.get_value(),
             Ui=model.Ui.get_value(),
             Wf=model.Wf.get_value(),
             Uf=model.Uf.get_value(),
             Wo=model.Wo.get_value(),
             Uo=model.Uo.get_value(),
             Wc=model.Wc.get_value(),
             Uc=model.Uc.get_value(),
             User_vector=model.User_vector.get_value(),
             Poi_vector=model.Poi_vector.get_value())
    print "Saved model parameters to %s." % outfile


def load_model_parameters_lstm(path, modelClass=LSTMtheano):
    npzfile = np.load(path)
    Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector = npzfile["Wi"], npzfile["Ui"], npzfile["Wf"], npzfile[
        "Uf"], npzfile["Wo"], npzfile["Uo"], npzfile["Wc"], npzfile["Uc"], npzfile["User_vector"], npzfile["Poi_vector"]
    hidden_dim = Wi.shape[0]
    print "Building model, model from %s with hidden_dim=%d " % (path, hidden_dim)
    sys.stdout.flush()
    num_user = read_gtd_users()
    num_poi = read_gtd_pois()
    model = modelClass(hidden_dim=hidden_dim, num_user=num_user, num_poi=num_poi)
    model.Wi.set_value(Wi)
    model.Ui.set_value(Ui)
    model.Wf.set_value(Wf)
    model.Uf.set_value(Uf)
    model.Wo.set_value(Wo)
    model.Uo.set_value(Uo)
    model.Wc.set_value(Wc)
    model.Uc.set_value(Uc)
    model.User_vector.set_value(User_vector)
    model.Poi_vector.set_value(Poi_vector)
    return model


def save_model_parameters_rnn(model, outfile):
    np.savez(outfile,
             M=model.M.get_value(),
             C=model.C.get_value(),
             # b=model.b.get_value(),
             User_vector=model.User_vector.get_value(),
             Poi_vector=model.Poi_vector.get_value())
    print "Saved model parameters to %s." % outfile


def load_model_parameters_rnn(path, modelClass=RNN):
    npzfile = np.load(path)
    M, C, User_vector, Poi_vector = npzfile["M"], npzfile["C"], npzfile["User_vector"], npzfile["Poi_vector"]
    hidden_dim = M.shape[0]
    user_dim = User_vector.shape[0]
    poi_dim = Poi_vector.shape[0]
    print "Building model, model from %s with hidden_dim=%d " % (path, hidden_dim)
    sys.stdout.flush()
    num_user = read_gtd_users()
    num_poi = read_gtd_pois()
    model = modelClass(user_dim=user_dim, poi_dim=poi_dim, hidden_dim=hidden_dim, num_user=num_user, num_poi=num_poi)
    model.M.set_value(M)
    model.C.set_value(C)
    model.User_vector.set_value(User_vector)
    model.Poi_vector.set_value(Poi_vector)
    return model


def load_model_parameters_rnn1(path, modelClass=RNNtheano):
    npzfile = np.load(path)
    M, C, User_vector, Poi_vector = npzfile["M"], npzfile["C"], npzfile["User_vector"], npzfile["Poi_vector"]
    hidden_dim = M.shape[0]
    user_dim = User_vector.shape[0]
    poi_dim = Poi_vector.shape[0]
    print "Building model, model from %s with hidden_dim=%d " % (path, hidden_dim)
    sys.stdout.flush()
    num_user = read_foursquare_users()
    num_poi = read_foursquare_pois()
    model = modelClass(user_dim=user_dim, poi_dim=poi_dim, hidden_dim=hidden_dim, num_user=num_user, num_poi=num_poi)
    model.M.set_value(M)
    model.C.set_value(C)
    model.User_vector.set_value(User_vector)
    model.Poi_vector.set_value(Poi_vector)
    return model


def load_model_parameters_lstm1(path, modelClass=LSTM_neg):
    npzfile = np.load(path)
    Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector = npzfile["Wi"], npzfile["Ui"], npzfile["Wf"], npzfile[
        "Uf"], npzfile["Wo"], npzfile["Uo"], npzfile["Wc"], npzfile["Uc"], npzfile["User_vector"], npzfile["Poi_vector"]
    hidden_dim = Wi.shape[0]
    print "Building model, model from %s with hidden_dim=%d " % (path, hidden_dim)
    sys.stdout.flush()
    num_user = read_gtd_users()
    num_poi = read_gtd_pois()
    model = modelClass(hidden_dim=hidden_dim, num_user=num_user, num_poi=num_poi)
    model.Wi.set_value(Wi)
    model.Ui.set_value(Ui)
    model.Wf.set_value(Wf)
    model.Uf.set_value(Uf)
    model.Wo.set_value(Wo)
    model.Uo.set_value(Uo)
    model.Wc.set_value(Wc)
    model.Uc.set_value(Uc)
    model.User_vector.set_value(User_vector)
    model.Poi_vector.set_value(Poi_vector)
    return model
