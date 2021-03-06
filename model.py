import theano
import lasagne
import theano.tensor as T
import numpy as np
import time
import properties
import utils 
import math
import random
# from pympler import tracker

from layers import LSTM, HiddenLayer, HiddenLayerDropout, FullConnectLayer

floatX = theano.config.floatX

class Model():
    
    def __init__(self, word_vectors, hidden_sizes=[300, 100, 2], dropout_rate=0.5, \
                batch_size=50, epochs=20, patience=10, learning_rate=0.13):
        self.word_vectors = word_vectors
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.patience_frq = 2
        self.learning_rate = learning_rate

    def train(self, train_data, dev_data, test_data, maxlen):
        # tr = tracker.SummaryTracker()
        rng = np.random.RandomState(3435)
        train_x, train_y = self.shared_dataset(train_data)
        dev_x, dev_y = self.shared_dataset(dev_data)
        test_x, test_y = self.shared_dataset(test_data)
        test_len = len(test_data[0])
        n_train_batches = len(train_data[0]) // self.batch_size
        n_val_batches = len(dev_data[0]) // self.batch_size
        n_test_batches = test_len // self.batch_size
        input_width = self.hidden_sizes[0]
        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()
        Words = theano.shared(value=self.word_vectors, name="Words", borrow=True)
        layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((self.batch_size, maxlen, input_width))
        lstm = LSTM(dim=input_width, batch_size=self.batch_size, number_step=maxlen)
        leyer0_output = lstm.feed_foward(layer0_input)
        lstm.mean_pooling_input(leyer0_output)
        hidden_sizes = [self.hidden_sizes[0], self.hidden_sizes[0]]
        hidden_layer = HiddenLayer(rng, hidden_sizes=hidden_sizes, input_vectors=lstm.output, activation=utils.Tanh, name="Hidden_Tanh") 
        hidden_layer.predict()
        hidden_layer_relu = HiddenLayer(rng, hidden_sizes=hidden_sizes, input_vectors=hidden_layer.output)
        hidden_layer_relu.predict()
        # hidden_layer_dropout = HiddenLayerDropout(rng, hidden_sizes=self.hidden_sizes[:2], input_vectors=lstm.output, W=hidden_layer.W, b=hidden_layer.b)
        full_connect = FullConnectLayer(rng, layers_size=[self.hidden_sizes[0], self.hidden_sizes[-1]], input_vector=hidden_layer_relu.output)
        full_connect.predict()

        cost = full_connect.negative_log_likelihood(y)
        
        params = lstm.params + hidden_layer.params + hidden_layer_relu.params + full_connect.params
        # params = hidden_layer.params + hidden_layer_relu.params + full_connect.params
        params_length = len(params)
        #init value for e_grad time 0, e_delta time 0 and delta at time 0
        e_grad, e_delta_prev, delta = self.init_hyper_values(params_length)
        # e_grad_d, e_delta_prev_d, delta_d = self.init_hyper_values(params_length, name="D")
        #apply gradient
        grads = T.grad(cost, params)
        #dropout hidden layer
        # hidden_layer_dropout.dropout()
        # hidden_layer_dropout.predict()
        # full_connect.setInput(hidden_layer_dropout.output)
        # full_connect.predict()
        # cost_d = full_connect.negative_log_likelihood(y)
        #apply gradient to cost_d
        # grads_d = T.grad(cost_d, params)
        e_grad, e_delta_prev, delta = self.adadelta(grads, e_grad, e_delta_prev)
        # e_grad_d, e_delta_prev_d, delta_d = self.adadelta(grads_d, e_grad_d, e_delta_prev_d, delta_d)
        grads = delta
        # grad_d = delta_d
        updates = [(p, p - d) for p, d in zip(params, grads)]
        # updates = [(p, p - self.learning_rate * d) for p, d in zip(params, grads)]
        train_model = theano.function([index], cost, updates=updates, givens={
            x: train_x[(index * self.batch_size):((index + 1) * self.batch_size)],
            y: train_y[(index * self.batch_size):((index + 1) * self.batch_size)]
        })
        val_model = theano.function([index], full_connect.errors(y), givens={
            x: dev_x[index * self.batch_size: (index + 1) * self.batch_size],
            y: dev_y[index * self.batch_size: (index + 1) * self.batch_size],
        })
        test_model = theano.function(inputs=[index], outputs=full_connect.errors(y), givens={
            x: test_x[index * self.batch_size: (index + 1) * self.batch_size],
            y: test_y[index * self.batch_size: (index + 1) * self.batch_size]
        })
        validation_frequency = min(n_train_batches, self.patience // 2)
        val_batch_lost = 1.
        best_batch_lost = 1.
        best_test_lost = 1.
        stop_count = 0
        epoch = 0
        done_loop = False
        current_time_step = 0
        improve_threshold = 0.995
        iter_list = range(n_train_batches)
        while(epoch < self.epochs and done_loop is not True):
            epoch_cost_train = 0.
            epoch += 1
            batch_train = 0
            print("Start epoch: %i" % epoch)
            start = time.time()
            random.shuffle(iter_list)
            for mini_batch, m_b_i in zip(iter_list, xrange(n_train_batches)):
                current_time_step = (epoch - 1) * n_train_batches + m_b_i
                epoch_cost_train += train_model(mini_batch)
                batch_train += 1
                if (current_time_step + 1) % validation_frequency == 0:
                    val_losses = [val_model(i) for i in xrange(n_val_batches)]
                    val_losses = np.array(val_losses)
                    val_batch_lost = np.mean(val_losses)
                    if val_batch_lost < best_batch_lost:
                        if best_batch_lost * improve_threshold > val_batch_lost:
                            self.patience = max(self.patience, current_time_step * self.patience_frq)
                            best_batch_lost = val_batch_lost
                            # test it on the test set
                            test_losses = [
                                test_model(i)
                                for i in range(n_test_batches)
                            ]
                            current_test_lost = np.mean(test_losses)
                            print(('epoch %i minibatch %i test accuracy of %i example is: %.5f') % (epoch, m_b_i, test_len, (1 - current_test_lost) * 100.))
                            if best_test_lost > current_test_lost:
                                best_test_lost = current_test_lost
                if self.patience <= current_time_step:
                    print(self.patience)
                    done_loop = True
                    break
            print('epoch: %i, training time: %.2f secs; with avg cost: %.5f' % (epoch, time.time() - start, epoch_cost_train / batch_train))
        print('Best test accuracy is: %.5f' % (1 - best_test_lost))
        utils.save_layer_params(lstm, 'lstm')
        utils.save_layer_params(hidden_layer, 'hidden_lstm')
        utils.save_layer_params(hidden_layer_relu, 'hidden_relu_lstm')
        utils.save_layer_params(full_connect, 'full_connect_lstm')
        return lstm.params

    def build_test_model(self, data):
        rng = np.random.RandomState(3435)
        lstm_params, hidden_params, hidden_relu_params, full_connect_params = self.load_trained_params()
        data_x, data_y, maxlen = data
        test_len = len(data_x)
        n_test_batches = test_len // self.batch_size
        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()
        Words = theano.shared(value=self.word_vectors, name="Words", borrow=True)
        input_width = self.hidden_sizes[0]
        layer0_input = T.cast(Words[T.cast(x.flatten(), dtype="int32")], dtype=floatX).reshape((self.batch_size, maxlen, input_width))
        lstm = LSTM(dim=input_width, batch_size=self.batch_size, number_step=maxlen, params=lstm_params)
        layer0_input = lstm.feed_foward(layer0_input)
        lstm.mean_pooling_input(layer0_input)
        hidden_sizes = [self.hidden_sizes[0], self.hidden_sizes[0]]
        hidden_layer = HiddenLayer(rng, hidden_sizes=hidden_sizes, input_vectors=lstm.output, activation=utils.Tanh, name="Hidden_Tanh", W=hidden_params[0], b=hidden_params[1]) 
        hidden_layer.predict()
        hidden_layer_relu = HiddenLayer(rng, hidden_sizes=hidden_sizes, input_vectors=hidden_layer.output, W=hidden_relu_params[0], b=hidden_relu_params[1])
        hidden_layer_relu.predict()
        # hidden_layer_dropout = HiddenLayerDropout(rng, hidden_sizes=self.hidden_sizes[:2], input_vectors=lstm.output, W=hidden_layer.W, b=hidden_layer.b)
        full_connect = FullConnectLayer(rng, layers_size=[self.hidden_sizes[0], self.hidden_sizes[-1]], 
                                        input_vector=hidden_layer_relu.output, W=full_connect_params[0], b=full_connect_params[1])
        full_connect.predict()
        test_data_x = theano.shared(np.asarray(data_x, dtype=floatX), borrow=True)
        test_data_y = theano.shared(np.asarray(data_y, dtype='int32'), borrow=True)
      
        errors = 0.
        if test_len == 1:
            test_model = theano.function([index],outputs=full_connect.get_predict(), on_unused_input='ignore', givens={
                x: test_data_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: test_data_y[index * self.batch_size: (index + 1) * self.batch_size]
            })
            index = 0
            avg_errors = test_model(index)
        else:
            test_model = theano.function([index], outputs=full_connect.errors(y), givens={
                x: test_data_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: test_data_y[index * self.batch_size: (index + 1) * self.batch_size]
            })
            for i in xrange(n_test_batches):
                errors += test_model(i)
            avg_errors = errors / n_test_batches
        return avg_errors

    def load_trained_params(self):
        lstm = utils.load_file('lstm.txt')
        hidden_lstm = utils.load_file('hidden_lstm.txt')
        hidden_relu_lstm = utils.load_file('hidden_relu_lstm.txt')
        full_connect_lstm = utils.load_file('full_connect_lstm.txt')
        return lstm, hidden_lstm, hidden_relu_lstm, full_connect_lstm

    def shared_dataset(self, data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(
            data_x, dtype=floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(
            data_y, dtype=floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    

    def init_hyper_values(self, length, name="N"):
        # e_grad = theano.shared(np.zeros(length, dtype=floatX), name="e_grad" + name)
        # e_delta = theano.shared(np.zeros(length, dtype=floatX), name="e_delta" + name)
        # delta = theano.shared(np.zeros(length, dtype=floatX), name="delta" + name)
        e_grad = np.zeros(length, dtype=floatX)
        e_delta = np.zeros(length, dtype=floatX)
        delta = np.zeros(length, dtype=floatX)
        return e_grad, e_delta, delta

    #e_delta_prev is e of delta of two previous step
    def adadelta(self, grads, e_g_prev, e_delta):
        #calculate e value for grad from e g previous and current grad
        e_grad = self.average_value(e_g_prev, grads)
        #calculate rms for grad
        rms_g = self.RMS(e_grad)
        #rms0 = sqrt(epsilon)
        rms_e_del_prev = self.RMS(e_delta)
        #delta of current time
        delta = [rd / rg * g for rd, rg, g in zip(rms_e_del_prev, rms_g, grads)]
        #e value of delta of time t
        e_delta_1 = self.average_value(e_delta, delta)
        
        return e_grad, e_delta_1, delta
    
    def rms_prop(self, grads, e_g_prev):
        e_grad = self.average_value(e_g_prev, grads)
        rms_g = self.RMS(e_grad)
        delta = delta = self.cal_delta(rms_g, grads)
        return e_grad, delta

    def adagrad(self, grads):
        G = self.sum_diag(grads)
        rms_g = self.RMS(G)
        delta = self.cal_delta(rms_g, grads)

    def momentum(self, grads, v_prev):
        #velocity is delta
        velocity = self.get_velocity(grads, v_prev)
        return velocity

    def get_velocity(self, grads, v_prev):
        return [v + properties.n_gamma + properties.learning_rate * g for g, v in zip(grads, v_prev)]

    def sum_diag(self, grads):
        return [T.sum(T.nlinalg.ExtractDiag(g)) for g in grads]

    def average_value(self, E_prev, grads):
        # grads_ = [T.cast(i, floatX) for i in grads]
        # return E_prev * properties.gamma + (1 - properties.gamma) * grads_
        f_gm = lasagne.utils.floatX(properties.gamma)
        return [e * f_gm + (lasagne.utils.floatX(1.) - f_gm) * (g**2.) for e, g in  zip(E_prev, grads)]

    def RMS(self, values):
        return [T.sqrt(e + lasagne.utils.floatX(properties.epsilon)) for e in  values]
        # return T.sqrt(values + properties.epsilon)
    
    def cal_delta(self, denominator, grads):
        return [properties.learning_rate / rms * g for rms, g in zip(denominator, grads)]
