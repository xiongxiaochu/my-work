import numpy as np
import theano as theano
import theano.tensor as T
from Evaluate import *


class RNNtheano:
    def __init__(self, user_dim=30, poi_dim=30, hidden_dim=13, top_K=10, regu_para=0.0001,
                 num_user=5000, num_poi=100000):
        # Assign instance variables
        self.user_dim = user_dim
        self.poi_dim = poi_dim
        self.hidden_dim = hidden_dim
        self.top_K = top_K
        # the number of negative samples
        self.regu_para = regu_para
        self.num_user = num_user
        self.num_poi = num_poi

        # Randomly initialize the network parameters
        M = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / poi_dim), (hidden_dim, poi_dim))
        C = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, hidden_dim))
        User_vector = np.random.random_sample((user_dim, num_user)) - 0.5
        Poi_vector = np.random.random_sample((poi_dim, num_poi)) - 0.5

        # Theano: Created shared variables
        self.M = theano.shared(name='M', value=M.astype(theano.config.floatX))
        self.C = theano.shared(name='C', value=C.astype(theano.config.floatX))
        self.User_vector = theano.shared(name='User_vector', value=User_vector.astype(theano.config.floatX))
        self.Poi_vector = theano.shared(name='Poi_vector', value=Poi_vector.astype(theano.config.floatX))

        # We store the theano graph here
        # self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        M, C, User_vector, Poi_vector = self.M, self.C, self.User_vector, self.Poi_vector
        x = T.ivector('positive_samples')
        y = T.imatrix('negative_samples')
        u = T.iscalar('u')
        # index is a vector to index the position of output probability matrix
        # index = T.ivector('index')

        # forward propagation step for calculating loss function
        def cal_loss_step(x_t, y_t, h_t_prev, u, M, C, User_vector, Poi_vector):
            # correct poi-embedding
            x_c = Poi_vector[:, x_t]
            # user-embedding layer
            u_e = User_vector[:, u]
            h_t = T.nnet.sigmoid(M.dot(x_c) + C.dot(h_t_prev))
            # h_t = M.dot(x_e) + C.dot(h_t_prev)
            correct_prob = (h_t + u_e).dot(x_c)

            def step(y_t1, correct_prob, h_t_prev, u_e, M, C, User_vector, Poi_vector):
                # negative poi_embedding
                x_e = Poi_vector[:, y_t1]
                h_t1 = T.nnet.sigmoid(M.dot(x_e) + C.dot(h_t_prev))
                negative_prob = (h_t1 + u_e).dot(x_e)
                loss = T.log(1 + T.exp(-(correct_prob - negative_prob)))
                return loss
            neg_loss, _ = theano.scan(step,
                                      sequences=[y_t],
                                      outputs_info=[None],
                                      non_sequences=[correct_prob, h_t_prev, u_e, M, C, User_vector, Poi_vector])
            return [T.sum(neg_loss), h_t]

        [loss_out, h], updates = theano.scan(cal_loss_step,
                                             sequences=[x, y],
                                             outputs_info=[None,
                                                           dict(initial=T.zeros(self.hidden_dim))],
                                             non_sequences=[u, M, C, User_vector, Poi_vector])

        self.forward = theano.function([x, y, u], loss_out)

        matrix_norm = M.norm(2) + C.norm(2) + User_vector.norm(2) + Poi_vector.norm(2)

        # final_loss = medium_loss + self.regu_para / 2 * matrix_norm

        final_loss = T.sum(loss_out) + self.regu_para / 2 * matrix_norm

        # forward propagation for predict
        samples = T.ivector('test_samples')

        def forward_step(x_t, h_t_prev, u, M, C, User_vector, Poi_vector):
            # poi-embedding layer
            x_e = Poi_vector[:, x_t]
            # user-embedding layer
            u_e = User_vector[:, u]
            h_t = T.nnet.sigmoid(M.dot(x_e) + C.dot(h_t_prev))
            # h_t = M.dot(x_e) + C.dot(h_t_prev)
            o = (h_t + u_e).dot(Poi_vector)
            out_t = T.nnet.softmax(o)[0]
            return [out_t, h_t]
        [out, h], updates = theano.scan(forward_step,
                                        sequences=samples,
                                        outputs_info=[None,
                                                      dict(initial=T.zeros(self.hidden_dim))],
                                        non_sequences=[u, M, C, User_vector, Poi_vector])

        self.forward_propagation = theano.function([samples, u], out)
        # Gradients
        dM = T.grad(final_loss, M)
        dC = T.grad(final_loss, C)
        dUser_vector = T.grad(final_loss, User_vector)
        dPoi_vector = T.grad(final_loss, Poi_vector)

        # Assign functions
        self.cal_loss_function = theano.function([x, y, u], final_loss)
        # self.check_M = theano.function([], M)
        # self.check_C = theano.function([], C)
        # self.check_User_vector = theano.function([], User_vector)
        # self.check_Poi_vector = theano.function([], Poi_vector)

        # SGD
        learning_rate = T.scalar('learning_rate')
        batch_size = T.scalar('batch_size')
        self.sdg_step = theano.function([x, y, u, learning_rate, batch_size], [],
                                        updates=[(self.M, self.M - learning_rate * dM / batch_size),
                                                 (self.C, self.C - learning_rate * dC / batch_size),
                                                 (self.User_vector, self.User_vector - learning_rate * dUser_vector / batch_size),
                                                 (self.Poi_vector, self.Poi_vector - learning_rate * dPoi_vector / batch_size)])

    def predict(self, x, u):
        # perform forward propagation and return the probability of each poi
        out = self.forward_propagation(x, u)
        predict = np.zeros((len(x), self.top_K))
        for i in np.arange(out.shape[0]):
            predict[i] = get_topK(out[i], self.top_K)
        # predict is a matrix of shape len(x)*top_K
        return predict
