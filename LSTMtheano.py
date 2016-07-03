from __future__ import print_function
import numpy as np
import theano as theano
import theano.tensor as T
from Evaluate import *
import time as time
# theano.config.compute_test_value = 'off'

class LSTMtheano:
    def __init__(self, user_dim=13, poi_dim=13, hidden_dim=13, top_K=500, regu_para=0.01,
                 num_user=5000, num_poi=100000):
        # Assign instance variables
        self.user_dim = user_dim
        self.poi_dim = poi_dim
        self.hidden_dim = hidden_dim
        self.top_K = top_K
        # the number of negative samples
        # self.negative = negative
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
        User_vector = np.random.random_sample((user_dim, num_user)) - 0.5
        Poi_vector = np.random.random_sample((poi_dim, num_poi)) - 0.5

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
        # index is a vector to index the position of output probability matrix
        index = T.ivector('index')

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
            o = T.nnet.sigmoid((h_t + u_e).dot(Poi_vector))
            out_t = T.nnet.softmax(o)[0]
            return [out_t, c_t, h_t]

        # print time.time()
        [out, c, h], updates = theano.scan(forward_prop,
                                           sequences=x,
                                           outputs_info=[None,
                                                         dict(initial=T.zeros(self.hidden_dim)),
                                                         dict(initial=T.zeros(self.hidden_dim))],
                                           non_sequences=[u, Wi, Ui, Wf, Uf, Wo, Uo, Wc, Uc, User_vector, Poi_vector])
        error = T.sum(T.nnet.categorical_crossentropy(out, y))

        # final_out = T.nnet.softmax(out)
        # print time.time()
        # def loss_function(y_t, i):
        #     correct_prob = out[i, y_t]
        
        #     pos = np.random.randint(0, len(self.poi_set))
        #     while pos == y_t:
        #         pos = np.random.randint(0, len(self.poi_set))
        #     neg_prob = out[i, pos]
        #     L = T.log(1 + T.exp(-(correct_prob - neg_prob)))
        #     return L
        # print time.time()
        # loss, updates = theano.scan(loss_function, sequences=[y, index])
        # print time.time()
        # medium_loss = loss.sum()
        matrix_norm = Wi.norm(2) + Ui.norm(2) + Wf.norm(2) + Uf.norm(2) + Wo.norm(2) + Uo.norm(2) + Wc.norm(2) \
                      + Uc.norm(2) + User_vector.norm(2) + Poi_vector.norm(2)
        # final_loss = medium_loss + self.regu_para / 2 * matrix_norm
        # print time.time()

        final_loss = error + (self.regu_para / 2) * matrix_norm

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

        # Assign functions
        # def detect_nan(i, node, fn):
        #     for output in fn.outputs:
        #         if (not isinstance(output[0], numpy.random.RandomState) and
        #             numpy.isnan(output[0]).any()): 
        #             print('*** NaN detected ***')
        #             theano.printing.debugprint(node)
        #             print('Inputs : %s' % [input[0] for input in fn.inputs])
        #             print('Outputs: %s' % [output[0] for output in fn.outputs])
        #             break

        # bug_log = open('../bug_log.txt','w')
        # def inspect_inputs(i, node, fn):
        #     print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],end='')
        # def inspect_outputs(i, node, fn):
        #     print(" output(s) value(s):", [output[0] for output in fn.outputs])
        # mode = theano.compile.MonitorMode(post_func=detect_nan).excluding('local_elemwise_fusion', 'inplace')
        self.forward_propagation = theano.function([x, u], out)

        self.cal_loss_function = theano.function([x, y, u], final_loss)
        # self.bptt = theano.function([x, y, u], [dWi, dUi, dWf, dUf, dWo, dUo, dWc, dUc, dUser_vector, dPoi_vector])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sdg_step = theano.function([x, y, u, learning_rate], [],
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
