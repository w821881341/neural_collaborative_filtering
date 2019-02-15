'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np

# import os
# os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate2 import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
from keras.utils.visualize_util import plot


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)


def get_model(train_matrix, num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables

    user_input = Input(shape=(1,), dtype='float32', name='user_input')
    item_input = Input(shape=(1,), dtype='float32', name='item_input')
    # train_matrix = Input(shape=(num_users,num_items), dtype='int32', name = 'train_matrix')
    # train_matrix_t = K.transpose(train_matrix)
    train_matrix_t = train_matrix.transpose()

    user_data = Flatten()(Embedding(num_users, num_items,
                                    weights=[train_matrix], input_length=1,
                                    name='embedding_user', trainable=False)(user_input))
    item_data = Flatten()(Embedding(num_items, num_users,
                                    weights=[train_matrix_t], input_length=1,
                                    name='embedding_item', trainable=False)(item_input))

    K.print_tensor(user_data)
    user_encoder = Sequential(name='user_encoder')
    user_encoder.add(Dense(layers[0], input_shape=(num_items,), activation='relu', name='user_encoder_layer_1'))
    # user_encoder.build((num_items,))

    user_decoder = Sequential(name='user_decoder')
    user_decoder.add(Dense(num_items, input_shape=(layers[0],), activation='relu', name='user_decoder_layer_1'))
    # user_decoder.build((layers[0],))

    item_encoder = Sequential(name='item_encoder')
    item_encoder.add(Dense(layers[0], input_shape=(num_users,), activation='relu', name='item_encoder_layer_1'))
    # item_encoder.build((num_users,))

    item_decoder = Sequential(name='item_decoder')
    item_decoder.add(Dense(num_users, input_shape=(layers[0],), activation='relu', name='item_decoder_layer_1'))
    # item_decoder.build((layers[0],))

    user_encoder_MLP = user_encoder(user_data)
    user_decoder_MLP = user_decoder(user_encoder_MLP)
    item_encoder_MLP = item_encoder(item_data)
    item_decoder_MLP = item_decoder(item_encoder_MLP)

    # The 0-th layer is the dot product of embedding layers
    vector = merge([user_encoder_MLP, item_encoder_MLP], mode='mul', name="fusion")

    MLP_layers = Sequential(name="MLP_layers")
    for idx in range(1, num_layer):
        MLP_layers.add(
            Dense(layers[idx], input_shape=(layers[idx - 1],), W_regularizer=l2(reg_layers[idx]), activation='relu',
                  name='layer%d' % idx))
    vector = MLP_layers(vector)
    # MLP_layers.build((layers[0],))

    predict_layer = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction', input_shape=(layers[-1],))
    predict_result = predict_layer(vector)

    user_cost = Lambda(
        lambda x: K.sum(K.square(x[0] - x[1]) * x[0], 1, keepdims=True) / (K.sum(x[0], 1, keepdims=True) + K.epsilon()),
        output_shape=(1,), name='user_reconstruct_cost')([user_data, user_decoder_MLP])
    item_cost = Lambda(
        lambda x: K.sum(K.square(x[0] - x[1]) * x[0], 1, keepdims=True) / (K.sum(x[0], 1, keepdims=True) + K.epsilon()),
        output_shape=(1,), name='item_reconstruct_cost')([item_data, item_decoder_MLP])
    model = Model(input=[user_input, item_input],
                  output=[predict_result, user_cost, item_cost])

    return model


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    cost_weight = args.cost_weight

    topK = 10
    evaluation_threads = 1
    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    train_matrix = np.array(train.toarray())
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(train_matrix, num_users, num_items, layers, reg_layers)
    plot(model, to_file='AutoDCF.png')

    cost_lambda = lambda y_true, y_pred: y_pred
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss=['binary_crossentropy', cost_lambda, cost_lambda],
                      loss_weights=[1.0-2*cost_weight, cost_weight, cost_weight])
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss=['binary_crossentropy', cost_lambda, cost_lambda],
                      loss_weights=[1.0-2*cost_weight, cost_weight, cost_weight])
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss=['binary_crossentropy', cost_lambda, cost_lambda],
                      loss_weights=[1.0-2*cost_weight, cost_weight, cost_weight])
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss=['binary_crossentropy', cost_lambda, cost_lambda],
                      loss_weights=[1.0-2*cost_weight, cost_weight, cost_weight])

        # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    # mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    # p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        user_input_array = np.array(user_input)
        item_input_array = np.array(item_input)
        label_array = np.array(labels)
        # Training
        hist = model.fit([user_input_array, item_input_array],  # input
                         [label_array, np.zeros_like(label_array, dtype=float),
                          np.zeros_like(label_array, dtype=float)],  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss, prediction_loss, user_cost, item_cost = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0], hist.history['prediction_loss'][0], hist.history['user_reconstruct_cost_loss'][0], hist.history['item_reconstruct_cost_loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, prediction_loss = %.4f, user_cost = %.4f, item_cost = %.4f, [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, prediction_loss, user_cost, item_cost, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))
