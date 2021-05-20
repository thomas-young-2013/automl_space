import os
import numpy as np
import pickle as pk
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Activation, Dense, Input, Subtract
from keras.layers import BatchNormalization
from keras.optimizers import SGD


class RankNetAdvisor(object):
    def __init__(self, algorithm_id):
        self.model = None
        self.n_candidate = None
        self.algorithm_id = algorithm_id
        self.model_dir = os.path.join('data', 'model_dir')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_path = os.path.join(self.model_dir, '%s_ranknet_model.pkl' % self.algorithm_id)

    def create_pairwise_data(self, X, y):
        X1, X2, labels = list(), list(), list()
        n_algo = y.shape[1]
        self.n_candidate = n_algo

        for _X, _y in zip(X, y):
            if np.isnan(_X).any():
                continue
            meta_vec = _X
            for i in range(n_algo):
                for j in range(i + 1, n_algo):
                    if (_y[i] == -1) or (_y[j] == -1):
                        continue

                    vector_i, vector_j = np.zeros(n_algo), np.zeros(n_algo)
                    vector_i[i] = 1
                    vector_j[j] = 1

                    meta_x1 = list(meta_vec.copy())
                    meta_x1.extend(vector_i.copy())

                    meta_x2 = list(meta_vec.copy())
                    meta_x2.extend(vector_j.copy())

                    X1.append(meta_x1)
                    X1.append(meta_x2)
                    X2.append(meta_x2)
                    X2.append(meta_x1)
                    _label = 1 if _y[i] > _y[j] else 0
                    labels.append(_label)
                    labels.append(1 - _label)
        return np.asarray(X1), np.asarray(X2), np.asarray(labels)

    @staticmethod
    def create_model(input_shape, hidden_layer_sizes, activation, solver):
        """
        Build Keras Ranker NN model (Ranknet / LambdaRank NN).
        """
        # Neural network structure
        hidden_layers = list()
        hidden_layers.append(BatchNormalization())

        for i in range(len(hidden_layer_sizes)):
            hidden_layers.append(
                Dense(hidden_layer_sizes[i], activation=activation[i], name=str(activation[i]) + '_layer' + str(i)))
        h0 = Dense(1, activation='linear', name='Identity_layer')
        input1 = Input(shape=(input_shape,), name='Input_layer1')
        input2 = Input(shape=(input_shape,), name='Input_layer2')
        x1 = input1
        x2 = input2
        for i in range(len(hidden_layer_sizes)):
            x1 = hidden_layers[i](x1)
            x2 = hidden_layers[i](x2)
        x1 = h0(x1)
        x2 = h0(x2)
        # Subtract layer
        subtracted = Subtract(name='Subtract_layer')([x1, x2])
        # sigmoid
        out = Activation('sigmoid', name='Activation_layer')(subtracted)
        # build model
        model = Model(inputs=[input1, input2], outputs=out)

        # categorical_hinge, binary_crossentropy
        # sgd = SGD(lr=0.3, momentum=0.9, decay=0.001, nesterov=False)
        model.compile(optimizer=solver, loss="categorical_hinge", metrics=['accuracy'])
        return model

    def fit(self, X, y, **kwargs):
        X1, X2, y_ = self.create_pairwise_data(X, y)
        print('Data shape', X1.shape)
        l1_size = kwargs.get('layer1_size', 256)
        l2_size = kwargs.get('layer2_size', 128)
        act_func = kwargs.get('activation', 'tanh')
        batch_size = kwargs.get('batch_size', 128)

        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
            else:
                self.model = self.create_model(X1.shape[1], hidden_layer_sizes=(l1_size, l2_size,),
                                               activation=(act_func, act_func,),
                                               solver='adam')
                self.model.fit([X1, X2], y_, epochs=200, batch_size=batch_size)
                print("save model...")
                self.model.save(self.model_path)

    def predict(self, dataset_vec):
        assert (self.n_candidate != None), 'Please call <fit> first.'
        n_algo = self.n_candidate
        _X = list()
        for i in range(n_algo):
            vector_i = np.zeros(n_algo)
            vector_i[i] = 1
            item = list(dataset_vec.copy()) + list(vector_i)
            _X.append(item)
        X = np.asarray(_X)
        ranker_output = K.function([self.model.layers[0].input], [self.model.layers[-3].get_output_at(0)])
        return ranker_output([X])[0].ravel()

    def predict_ranking(self, dataset_vec, rank_objs=None):
        preds = self.predict(dataset_vec)
        print(preds)
        ranking = np.argsort(-np.array(preds))
        print(ranking)
        if rank_objs is None:
            return ranking
        else:
            return [rank_objs[idx] for idx in ranking]
