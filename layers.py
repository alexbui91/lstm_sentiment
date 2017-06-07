import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
import numpy as np
import utils
import properties

class LSTM(object):

    def __init__(self, dim, batch_size, number_step, params=None):
        self.dim = dim
        self.batch_size = batch_size
        self.number_step = number_step
        self.output = None
        if not params:
            self.init_params()
        else:
            self.set_params(params)

    def init_params(self):
        Wi_values = utils.ortho_weight(self.dim)
        self.Wi = theano.shared(Wi_values, name="LSTM_Wi")
        Wf_values = utils.ortho_weight(self.dim)
        self.Wf = theano.shared(Wf_values, name="LSTM_Wf")
        Wc_values = utils.ortho_weight(self.dim)
        self.Wc = theano.shared(Wc_values, name="LSTM_Wc")
        Wo_values = utils.ortho_weight(self.dim)
        self.Wo = theano.shared(Wo_values, name="LSTM_Wo")
        Ui_values = utils.ortho_weight(self.dim)
        self.Ui = theano.shared(Ui_values, name="LSTM_Ui")
        Uf_values = utils.ortho_weight(self.dim)
        self.Uf = theano.shared(Uf_values, name="LSTM_Uf")
        Uc_values = utils.ortho_weight(self.dim)
        self.Uc = theano.shared(Uc_values, name="LSTM_Uc")
        Uo_values = utils.ortho_weight(self.dim)
        self.Uo = theano.shared(Uo_values, name="LSTM_Uo")
        b_values = np.zeros((self.dim,), dtype=theano.config.floatX)
        self.bi = theano.shared(b_values, name="LSTM_bi")
        self.bf = theano.shared(b_values, name="LSTM_bf")
        self.bc = theano.shared(b_values, name="LSTM_bc")
        self.bo = theano.shared(b_values, name="LSTM_bo")
        self.params = [self.Wi, self.Ui, self.bi, self.Wf, self.Uf, self.bf, self.Wc, self.Uc, self.bc, self.Wo, self.Uo, self.bo]
    
    def set_params(self, params):
        if len(params) is 12:
            self.params = params
            self.Wi = params[0]
            self.Ui = params[1]
            self.bi = params[2]
            self.Wf = params[3]
            self.Uf = params[4]
            self.bf = params[5]
            self.Wc = params[6]
            self.Uc = params[7]
            self.bc = params[8]
            self.Wo = params[9]
            self.Uo = params[10]
            self.bo = params[11]

    def get_params(self):
        return self.params
    
    def feed_foward(self, layer_input):
        #xt, h(t-1), c(t-1)
        #scan over sequence * batch size * width
        X_shuffled = T.cast(layer_input.dimshuffle(1,0,2), theano.config.floatX)
        def step(x, h_, C_):
            i = T.nnet.sigmoid(T.dot(x, self.Wi) + T.dot(h_, self.Ui) + self.bi)
            f = T.nnet.sigmoid(T.dot(x, self.Wf) + T.dot(h_, self.Uf) + self.bf)
            c = T.tanh(T.dot(x, self.Wc) + T.dot(h_, self.Uc) + self.bc)
            o = T.nnet.sigmoid(T.dot(x, self.Wo) + T.dot(h_, self.Uo) + self.bo)
            C = c * i + f * C_
            h = o * T.tanh(C)
            return h, C
        results, updates = theano.scan(step, outputs_info=[T.alloc(np.asarray((0.), dtype=theano.config.floatX), self.batch_size, self.dim),
                                                            T.alloc(np.asarray((0.), dtype=theano.config.floatX), self.batch_size, self.dim)],
                                            sequences=[X_shuffled],
                                            name="LSTM_iteration",
                                            n_steps=self.number_step)
        #get h after perform LSTMs
        return results[0]
    
    def mean_pooling_input(self, layer_input):
        #axis = 0 is x(col), = 1 is y (row),
        self.output = T.mean(layer_input, axis=0)

class NetworkLayer(object):

    def initHyperParamsFromValue(self, W, b, name="values"):
        W_param = theano.shared(value=W, borrow=True, name=("W_" + name))
        b_param = theano.shared(value=b, borrow=True, name=("b_" + name))
        self.W = W_param
        self.b = b_param
        self.params = [self.W, self.b]

class ConvolutionLayer(NetworkLayer):
    def __init__(self, rng=None, filter_shape=None, input_shape=None, poolsize=(2, 2), non_linear="tanh", name="Conv", W=None, b=None):
        #filter_shape = number of kenel, channel, height, width
        #input_shape = batch_size, channel, height, width
        #poolsize = height_pool x width_pool (if width = 1 mean select all vector word)
        assert input_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.non_linear = non_linear
        self.poolsize = poolsize
        self.rng = rng
        self.name = name
        if not W or not b:
            self.initHyperParams()
        else:
            self.initHyperParamsFromValue(W, b, name=name)

    def initHyperParams(self):
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(self.filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) / np.prod(self.poolsize))
        # initialize weights with random weights
        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=self.filter_shape), dtype=theano.config.floatX), name="W_" + self.name)
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape), dtype=theano.config.floatX), name="W_" + self.name)
        b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name="b_conv" + self.name)
        self.params = [self.W, self.b]

    def predict(self, new_data):
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=self.input_shape)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = pool.pool_2d(input=conv_out_tanh, ws=self.poolsize, ignore_border=True)
        if self.non_linear == "relu":
            conv_out_tanh = utils.ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = pool.pool_2d(input=conv_out_tanh, ws=self.poolsize, ignore_border=True)
        else:
            pooled_out = pool.pool_2d(input=conv_out, ws=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output

class HiddenLayer(NetworkLayer):
    
    def __init__(self, rng=None, activation=None, hidden_sizes=None, W=None, b=None, input_vectors=None, name="Hidden"):
        self.rng = rng
        if not activation:
            activation = utils.ReLU
        self.activation = activation
        self.input = input_vectors
        self.n_in = hidden_sizes[0]
        self.n_out = hidden_sizes[-1]
        self.W = W
        self.b = b
        self.name = name
        self.output = None
        if not self.W or not self.b:
            self.init_params()
        else: 
            self.initHyperParamsFromValue(W, b, 'hidden_layer')
    
    def init_params(self):
        if not self.activation or self.activation.func_name == "ReLU":
            #standard normal distribution when muy = 0 and Matrix = I
            W_values = np.asarray(0.01 * self.rng.standard_normal(size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
        else:
            # p distribution = 1/(high - low)
            W_values = np.asarray(self.rng.uniform(low=-np.sqrt(6. / (self.n_in + self.n_out)), high=np.sqrt(6. / (self.n_in + self.n_out)), size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W_' + self.name)
        b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_' + self.name)
        self.params = [self.W, self.b]
    
    def predict(self):
        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (lin_output if self.activation is None else self.activation(lin_output))

class HiddenLayerDropout(HiddenLayer):
    
    def __init__(self, rng, activation=None, dropout_rate=0.5, hidden_sizes=None, W=None, b=None, input_vectors=None):
        super(self.__class__, self).__init__(rng, activation, hidden_sizes, W, b, input_vectors, name="Dropout")
        self.dropout_rate = dropout_rate
    
    def dropout(self):
        self.input_vectors = utils.dropout_from_layer(self.rng, self.input, self.dropout_rate)    

# full connect here is final layer, logistic regression => prob to calculate cost function y^ = softmax (W^T * input + b)
class FullConnectLayer(NetworkLayer):

    def __init__(self, rng=None, layers_size=None, input_vector=None, W=None, b=None):
        self.rng = rng
        self.layers_size = layers_size
        self.input_vector = input_vector
        if not W or not b:
            self.initHyperParams()
        else:
            self.initHyperParamsFromValue(W, b, 'full_connect')

    def initHyperParams(self):
        W_bound = np.sqrt(6. / (self.layers_size[0] + self.layers_size[1]))
        w_size = (self.layers_size[0], self.layers_size[1])
        self.W = theano.shared(
                    value=np.zeros(w_size, dtype=theano.config.floatX),
                    name='W_full')
        self.b = theano.shared(
                    value=np.zeros((self.layers_size[1],), dtype=theano.config.floatX),
                    name='b_full')
        self.params = [self.W, self.b]

    def setInput(self, inp):
        self.input_vector = inp

    def predict(self):
        y_p = self.predict_p()
        self.y_pred = T.argmax(y_p, axis=1)

    def predict_p(self):
        self.y_prob = T.nnet.softmax(T.dot(self.input_vector, self.W) + self.b)
        return self.y_prob

    def negative_log_likelihood(self, y):
        #only average over log of labeled propability
        return -T.mean(T.log(self.y_prob[T.arange(y.shape[0]), y] + properties.epsilon))

    def soft_negative_log_likelihood(self, y):
        #average of sum of all possible choices. y must be vector of final value respective to number of classifications
        #y shape is n batches * K classification
        return -T.mean(T.sum(T.log(self.y_prob) * y, axis=1))

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            #return total different values between target vs prediction of one batch
            #after that, calculate average over batch
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_predict(self):
        return T.mean(self.y_pred)