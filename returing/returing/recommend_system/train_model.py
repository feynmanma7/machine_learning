import keras.backend as K
from keras import Sequential, Model
from keras.layers import Embedding, Dense, \
    Lambda, Reshape, Bidirectional, LSTM, \
    Dropout, Flatten, Multiply, Input, \
    BatchNormalization, Permute, merge, dot, Activation, Concatenate
from keras.utils import np_utils
import numpy as np
np.random.seed(20170430)
import random

def sampling(arr_list, num_sample=10):
    return random.sample(arr_list, num_sample)

def file_generator(input_path, V=10, seq_length=5, batch_size=2):

    count = 0

    while True:
        seqs = []
        pos_doc = []

        neg_docs = []
        num_neg = 4
        for i in range(num_neg):
            neg_docs.append([])

        y = []

        with open(input_path, 'r') as f:

            for line in f:
                buf = line.split(',')

                #seqs.append(buf[:7])

                seqs.append(buf[:2])
                pos_doc.append(buf[2])

                for idx, neg_doc in enumerate(buf[3:7]):
                    neg_docs[idx].append(neg_doc)

                #y.append(np.random.randint(2))

                # sparse_binary_crossentropy
                y.append(0)

                count += 1

                if count % batch_size == 0:

                    """
                    yield np.array(seqs, dtype=np.float), \
                          np.array(y, dtype=np.float)
                    """

                    inputs = []

                    seqs = np.array(seqs, dtype=np.float)
                    inputs.append(seqs)

                    pos_doc = np.array(pos_doc, dtype=np.float)
                    inputs.append(pos_doc)

                    for i in range(num_neg):
                        neg_doc = np.array(neg_docs[i], dtype=np.float)
                        inputs.append(neg_doc)

                    y = np.array(y, dtype=np.float)

                    yield inputs, y

                    seqs = []
                    pos_doc = []

                    neg_docs = []
                    num_neg = 4
                    for i in range(num_neg):
                        neg_docs.append([])

                    y = []

                    count = 0


def data_generator(V=10, window_size=3, batch_size=2):

    while True:
        x = []
        y = []

        for _ in range(batch_size):
            # [0, 10)
            x_ = np.random.randint(0, 10, size=window_size*2)
            x.append(x_)

            y_ = np.random.randint(V)
            y_ = np_utils.to_categorical(y_, V)
            y.append(y_)

        yield np.array(x), np.array(y)
        x, y = [], []

def attention(inputs, seq_length):
    """
    Input: inputs (batch_size, seq_length, input_dim)
    Output: (batch_size, seq_length, input_dim)
    """

    #print('inputs.shape', inputs.shape)

    a = Permute((2, 1))(inputs)
    a = Dense(seq_length, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output = Multiply()([inputs, a_probs])
    return output


def bilstm(V=10,
                embedding_dim=5,
                seq_length=3):

    sku_seqs_input = Input(shape=(seq_length, ), name='sku_seqs_input')

    sku_seqs = Embedding(input_dim=V,
                        output_dim=embedding_dim,
                        #embeddings_initializer='glorot_uniform',
                        input_length=seq_length,
                        name='sku_seqs_emb')(sku_seqs_input)

    sku_seqs = Bidirectional(
        LSTM(embedding_dim, return_sequences=True))(sku_seqs)
    sku_seqs = attention(sku_seqs, seq_length)
    sku_seqs = BatchNormalization()(sku_seqs)
    sku_seqs = Dropout(0.5)(sku_seqs)
    sku_seqs = Flatten()(sku_seqs)
    output = Dense(1, activation='sigmoid')(sku_seqs)

    model = Model(inputs=sku_seqs_input, \
                  outputs=output)

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    return model


def bilstm_attention(sku_seqs_input, V=10, seq_length=1, embedding_dim=5):

    embedding_layer = Embedding(input_dim=V,
                         output_dim=embedding_dim,
                         # embeddings_initializer='glorot_uniform',
                         input_length=seq_length,
                         name='word_embedding_layer')

    sku_seqs = embedding_layer(sku_seqs_input)

    # num_params: 4 * ((x+y) * y + y), W_xh * X_t + W_hh * h_{t-1} + b_h
    sku_seqs = Bidirectional(LSTM(units=32, return_sequences=True),
                             merge_mode='ave')(sku_seqs)
    sku_seqs = attention(sku_seqs, seq_length)
    sku_seqs = BatchNormalization()(sku_seqs)
    sku_seqs = Dropout(0.5)(sku_seqs)
    sku_seqs = Flatten()(sku_seqs)
    output = Dense(8, activation='tanh')(sku_seqs)

    return output, embedding_layer


def embedding(
        seq_input,\
        input_dim=5,\
        output_dim=3,\
        input_length=3):

    output = Embedding(input_dim=input_dim,\
                    output_dim=output_dim,\
                    input_length=input_length)(seq_input)
    return output


def shared_embedding(
        weights,
        seq_input, \
        input_dim=5, \
        output_dim=3, \
        input_length=3):

    output = Embedding(input_dim=input_dim,\
                    output_dim=output_dim,\
                    input_length=input_length, \
                    trainable=False,
                    weights=weights)(seq_input)
    return output


def transforming(weights, seqs_input, V=10, seq_length=1, embedding_dim=5):

    seqs = shared_embedding(weights, \
                    seqs_input,\
                    input_dim=V,\
                    output_dim=embedding_dim,\
                    input_length=seq_length)

    seqs = Dense(units=8, activation='tanh')(seqs)
    seqs = Reshape((8, ))(seqs)

    return seqs


def compute_prob(query, docs):
    """
    Input:
        query: query transforming
        docs: list of doc(pos + negs) transforming

    Output:
        softmax of dot(query, doc)
    """
    q_doc_dot = [dot([query, doc], axes=1, normalize=True) for doc in docs]
    q_doc_dot = Concatenate(axis=1)(q_doc_dot)
    prob = Activation("softmax")(q_doc_dot)

    return prob

#def get_embedding_layer():


def dssm(seq_length = 5,
         num_neg = 6,
         V = 10,
         embedding_dim = 5):
    """
    Input:
    Output:
    """

    # transforming of query
    query_input = Input(shape=(seq_length,))
    query, embedding_layer = bilstm_attention(query_input, \
                             V = V,\
                             embedding_dim = embedding_dim,\
                             seq_length = seq_length)

    embedding_weights = get_weights(embedding_layer)

    # Using shared_embedding_weights
    # transforming of pos_doc
    pos_doc_input = Input(shape=(1,))
    pos_doc = transforming(
        embedding_weights,\
        pos_doc_input,\
        V = V,\
        embedding_dim = embedding_dim,\
        seq_length = 1)

    print('pos_doc.shape', pos_doc.shape)

    # transforming of neg_docs
    neg_docs_input = [Input(shape=(1,)) for _ in range(num_neg)]
    neg_docs = [transforming(
        embedding_weights,\
        neg_doc_input,\
        V = V,\
        embedding_dim = embedding_dim,\
        seq_length = 1) for neg_doc_input in neg_docs_input]
    print('neg_doc.shape', neg_docs[0].shape)

    # compute softmax probability
    prob = compute_prob(query, [pos_doc] + neg_docs)
    print('prob.shape', prob.shape)

    model = Model(inputs = [query_input, pos_doc_input] + neg_docs_input,\
                  outputs = prob)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
    return model

def get_weights(layer):
    weights = layer.get_weights()
    return weights

def train_model():
    V = 100 # vocabulary size
    embedding_dim = 32 # embedding_dim

    seq_length = 7 #

    """
    model = bilstm(V=V,
                    embedding_dim=embedding_dim,
                    seq_length=seq_length)
    """
    model = dssm(seq_length = 2,\
                 num_neg = 4,\
                 V = V,\
                 embedding_dim = embedding_dim)

    model.summary()

    model_weights_path = 'models/w2v.model'

    model.load_weights(model_weights_path)

    layer = model.get_layer("word_embedding_layer")
    print(layer.get_weights()[0].shape)

    return
    epochs = 100
    workers = 4
    num_samples = 100
    batch_size = 10
    steps_per_epoch = int(num_samples / batch_size)

    input_path = 'data/input.txt'
    model.fit_generator(file_generator(input_path=input_path,
                                       V=V,
                                       seq_length=seq_length,
                                       batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        workers=workers,
                        use_multiprocessing=True,
                        verbose=1
                        )


    model.save_weights(model_weights_path)



if __name__ == '__main__':
    train_model()