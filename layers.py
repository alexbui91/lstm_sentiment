import theano
import theano.tensor as T
import numpy as np
import utils

class LSTM(object):

    def __init__(self, dim, batch_size, number_step):
        self.dim = dim
        self.batch_size = batch_size
        self.number_step = number_step
        self.init_params()
        self.output = None
    
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
        self.params = [self.Wi, self.bi, self.Wf, self.bf, self.Wc, self.bc, self.Wo, self.bo]
    
    def get_params(self):
        return self.params
    
    def feed_foward(self, layer_input):
        #xt, h(t-1), c(t-1)
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
        self.output = T.mean(layer_input, axis=0)

class HiddenLayer(object):
    
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
        print(self.activation)
        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (lin_output if self.activation is None else self.activation(lin_output))

class HiddenLayerDropout(HiddenLayer):
    
    def __init__(self, rng, activation=None, dropout_rate=0.5, hidden_sizes=None, W=None, b=None, input_vectors=None):
        super(self.__class__, self).__init__(rng, activation, hidden_sizes, W, b, input_vectors, name="Dropout")
        self.dropout_rate = dropout_rate
    
    def dropout(self):
        self.input_vectors = utils.dropout_from_layer(self.rng, self.input, self.dropout_rate)    

# full connect here is final layer, logistic regression => prob to calculate cost function y^ = softmax (W^T * input + b)
class FullConnectLayer(object):

    def __init__(self, rng=None, layers_size=None, input_vector=None):
        self.rng = rng
        self.layers_size = layers_size
        self.input_vector = input_vector
        self.initHyperParams()

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
        return -T.mean(T.log(self.y_prob)[T.arange(y.shape[0]), y])

    def soft_negative_log_likelihood(self, y):
        return -T.mean(T.sum(T.log(self.y_pred) * y, axis=1))

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