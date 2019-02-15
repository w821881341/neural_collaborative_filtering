'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np

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
from new_evaluate import evaluate_model
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


def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables

    user_input = Input(shape=(num_items,), dtype='float32', name='user_input')
    item_input = Input(shape=(num_users,), dtype='float32', name='item_input')

    user_encoder = Dense(layers[0] , input_shape=(num_items,), activation='relu')
    user_decoder = Dense(num_items , input_shape=(layers[0],), activation='relu')
    item_encoder = Dense(layers[0], input_shape=(num_users,), activation='relu')
    item_decoder = Dense(num_users , input_shape=(layers[0],), activation='relu')

    user_encoder_MLP = user_encoder(user_input)
    user_decoder_MLP = user_decoder(user_encoder_MLP)
    item_encoder_MLP = item_encoder(user_input)
    item_decoder_MLP = item_decoder(user_encoder_MLP)


    # The 0-th layer is the dot product of embedding layers
    # vector = K.dot(user_encoder_MLP, item_encoder_MLP)
    vector = merge([user_encoder_MLP, item_encoder_MLP], mode='mul')
    # vector = T.dot(user_encoder_MLP,item_encoder_MLP)
    # MLP layers

    MLP_layers = Sequential()

    for idx in range(1, num_layer):
        MLP_layers.add(Dense(layers[idx],input_shape=(layers[idx-1],),W_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx))
    MLP_layers.build((layers[0],))
    vector = MLP_layers(vector)
    # Final prediction layer
    predict_layer = Sequential()
    predict_layer.add(Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction',input_shape=(layers[-1],)))
    predict_layer.build((layers[-1],))
    prediction = predict_layer(vector)
    model = Model(input=[user_input, item_input],
                  output=[user_decoder_MLP,item_decoder_MLP,prediction])

    return model


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    train_matrix = np.array(train.toarray())
    train_matrix_t = train_matrix.transpose()

    for (u, i) in train.keys():
        user_data = train_matrix[u, :]
        item_data = train_matrix_t[i, :]
        # positive instance
        user_input.append(user_data)
        item_input.append(item_data)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(user_data)
            item_input.append(train_matrix_t[j, :])
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

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
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
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss=['mse', 'mse', 'binary_crossentropy'], loss_weights=[0.25, 0.25, 0.5])
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss=['mse', 'mse', 'binary_crossentropy'], loss_weights=[0.25, 0.25, 0.5])
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss=['mse', 'mse', 'binary_crossentropy'], loss_weights=[0.25, 0.25, 0.5])
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss=['mse', 'mse', 'binary_crossentropy'], loss_weights=[0.25, 0.25, 0.5])

        # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(train_matrix,model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        user_input_array = np.array(user_input)
        item_input_array = np.array(item_input)
        # Training
        hist = model.fit([user_input_array, item_input_array],  # input
                         [user_input_array, item_input_array, np.array(labels)],  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(train_matrix, model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))
