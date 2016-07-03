import numpy as np
import theano as theano
import theano.tensor as T
from Evaluate import *


class lstm:
    def __init__(self, user_dim=20, poi_dim=20, hidden_dim=20, negative=10, top_K=20, regu_para=0.01,
                 num_user=5000, num_poi=100000):
        # Assign instance variables
        self.user_dim = user_dim
        self.poi_dim = poi_dim
        self.hidden_dim = hidden_dim
        self.top_K = top_K
        # the number of negative samples
        self.negative = negative
        self.regu_para = regu_para
        self.num_user = num_user
        self.num_poi = num_poi

        # Randomly initialize the network parameters
        Wi = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / poi_dim), (hidden_dim, poi_dim))
        Ui = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, hidden_dim))
        Wf = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / poi_dim), (hidden_dim, poi_dim))
        Uf = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, hidden_dim))
        Wo = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / poi_dim), (hidden_dim, poi_dim))
        Uo = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, hidden_dim))
        Wc = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / poi_dim), (hidden_dim, poi_dim))
        Uc = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, hidden_dim))
        User_vector = np.random.uniform(-np.sqrt(1.0 / user_dim), np.sqrt(1.0 / num_user),
                                        (user_dim, num_user))
        Poi_vector = np.random.uniform(-np.sqrt(1.0 / poi_dim), np.sqrt(1.0 / num_poi), (poi_dim, num_poi))

        # Theano: create shared variables
        self.Wi = theano.shared(name='Wi', value=Wi.astype(theano.config.floatX))
        self.Ui = theano.shared(name='Ui', value=Ui.astype(theano.config.floatX))
        self.Wf = theano.shared(name='Wf', value=Wf.astype(theano.config.floatX))
        self.Uf = theano.shared(name='Uf', value=Uf.astype(theano.config.floatX))
        self.Wo = theano.shared(name='Wo', value=Wo.astype(theano.config.floatX))
        self.Uo = theano.shared(name='Uo', value=Uo.astype(theano.config.floatX))
        self.Wc = theano.shared(name='Wc', value=Wc.astype(theano.config.floatX))
        self.Uc = theano.shared(name='Uc', value=Uc.astype(theano.config.floatX))
        self.User_vector = theano.shared(name='User_vector', value=User_vector.astype(theano.config.floatX))
        self.Poi_vector = theano.shared(name='Poi_vector', value=Poi_vector.astype(theano.config.floatX))

        # we store the Theano graph here
        # self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector = self.Wi, self.Ui, self.Wf, self.Uf, \
                                                                  self.Wo, self.Uo, self.Wc, self.Uc, self.User_vector, self.Poi_vector
        x = T.ivector('x')
        y = T.ivector('y')
        u = T.iscalar('u')

        def forward_step(x_t, y_t, u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector):
            # list: the first one poi is the correct poi, and the rest are the negative ones
            sampled = np.random.randint(low=0, high=self.num_poi, size=self.negative)
            neg_sample = theano.shared(name='neg_sample', value=sampled.astype(theano.config.floatX))

            def negative_step(poi, c_t_prev, h_t_prev, x_t, u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector):
                # input poi embedding
                x_e = Poi_vector[:, x_t]
                # target negative poi embedding
                x_c = Poi_vector[:, poi]
                # user-embedding layer
                u_e = User_vector[:, u]
                i_t = T.nnet.sigmoid(Wi.dot(x_e) + Ui.dot(h_t_prev))
                f_t = T.nnet.sigmoid(Wf.dot(x_e) + Uf.dot(h_t_prev))
                o_t = T.nnet.sigmoid(Wo.dot(x_e) + Uo.dot(h_t_prev))
                _c_t = T.nnet.sigmoid(Wc.dot(x_e) + Uc.dot(h_t_prev))
                c_t = f_t * c_t_prev + i_t * _c_t
                h_t = o_t * T.tanh(c_t)
                out_t = (h_t + u_e).dot(x_c)
                return [out_t, c_t, h_t]

            [negative_out, c1, h1], updates1 = theano.scan(fn=negative_step,
                                                           sequences=neg_sample,
                                                           outputs_info=[None,
                                                                         dict(initial=T.zeros(self.hidden_dim)),
                                                                         dict(initial=T.zeros(self.hidden_dim))],
                                                           non_sequences=[x_t, u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc,
                                                                          User_vector, Poi_vector])
            [correct_out, c2, h2], updates2 = theano.scan(fn=negative_step,
                                                          sequences=[y_t],
                                                          outputs_info=[None,
                                                                        dict(initial=T.zeros(self.hidden_dim)),
                                                                        dict(initial=T.zeros(self.hidden_dim))],
                                                          non_sequences=[x_t, u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc,
                                                                         User_vector, Poi_vector])
            return len(sampled) * correct_out - T.sum(negative_out)

        [o], updates = theano.scan(forward_step,
                                   sequences=[x, y],
                                   outputs_info=None,
                                   non_sequences=[u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector])

        # the output o is a vector in the size of time steps of the user
        neg_loss = T.sum(o)

        matrix_norm = Wi.norm(2) + Ui.norm(2) + Wf.norm(2) + Uf.norm(2) + Wo.norm(2) + Uo.norm(2) + Wc.norm(2) \
                      + Uc.norm(2) + User_vector.norm(2) + Poi_vector.norm(2)
        final_loss = neg_loss + self.regu_para / 2 * matrix_norm

        # final_loss = T.sum(T.nnet.categorical_crossentropy(out, y)) + self.regu_para / 2 * matrix_norm

        # Gradients
        dWi = T.grad(final_loss, Wi)
        dUi = T.grad(final_loss, Ui)
        dWf = T.grad(final_loss, Wf)
        dUf = T.grad(final_loss, Uf)
        dWo = T.grad(final_loss, Wo)
        dUo = T.grad(final_loss, Uo)
        dWc = T.grad(final_loss, Wc)
        dUc = T.grad(final_loss, Uc)
        dUser_vector = T.grad(final_loss, User_vector)
        dPoi_vector = T.grad(final_loss, Poi_vector)

        def forward_prop(x_t, c_t_prev, h_t_prev, u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector):
            # poi-embedding layer
            x_e = Poi_vector[:, x_t]
            # user-embedding layer
            u_e = User_vector[:, u]
            i_t = T.nnet.sigmoid(Wi.dot(x_e) + Ui.dot(h_t_prev))
            f_t = T.nnet.sigmoid(Wf.dot(x_e) + Uf.dot(h_t_prev))
            o_t = T.nnet.sigmoid(Wo.dot(x_e) + Uo.dot(h_t_prev))
            _c_t = T.nnet.sigmoid(Wc.dot(x_e) + Uc.dot(h_t_prev))
            c_t = f_t * c_t_prev + i_t * _c_t
            h_t = o_t * T.tanh(c_t)
            out_t = (h_t + u_e).dot(Poi_vector)
            return [out_t, c_t, h_t]

        [out, c, h], updates = theano.scan(forward_prop,
                                           sequences=x,
                                           outputs_info=[None,
                                                         dict(initial=T.zeros(self.hidden_dim)),
                                                         dict(initial=T.zeros(self.hidden_dim))],
                                           non_sequences=[u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector])

        # Assign functions
        self.forward_propagation = theano.function([x, u], out)
        self.cal_loss_function = theano.function([x, u], final_loss)
        # self.bptt = theano.function([x, y, u], [dWi, dUi, dWf, dUf, dWo, dUo, dWc, dUc, dUser_vector, dPoi_vector])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sdg_step = theano.function([x, u, learning_rate], [],
                                        updates=[(self.Wi, self.Wi - learning_rate * dWi),
                                                 (self.Ui, self.Ui - learning_rate * dUi),
                                                 (self.Wf, self.Wf - learning_rate * dWf),
                                                 (self.Uf, self.Uf - learning_rate * dUf),
                                                 (self.Wo, self.Wo - learning_rate * dWo),
                                                 (self.Uo, self.Uo - learning_rate * dUo),
                                                 (self.Wc, self.Wc - learning_rate * dWc),
                                                 (self.Uc, self.Uc - learning_rate * dUc),
                                                 (self.User_vector, self.User_vector - learning_rate * dUser_vector),
                                                 (self.Poi_vector, self.Poi_vector - learning_rate * dPoi_vector)])

    def predict(self, x, u):
        # perform forward propagation and return the probability of each poi
        out = self.forward_propagation(x, u)
        predict = np.zeros((len(x), self.top_K))
        for i in np.arange(out.shape[0]):
            predict[i] = get_topK(out[i], self.top_K)
        # predict is a matrix of shape len(x)*top_K
        return predict
