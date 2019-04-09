import keras.backend as K
from keras import Sequential, Model
from keras.layers import Embedding, Dense, \
    Lambda, Reshape, Bidirectional, LSTM, \
    Dropout, Flatten, Multiply, Input, \
    BatchNormalization, Permute, merge, dot, Activation, Concatenate
from keras.utils import np_utils
import numpy as np
np.random.seed(20170430)


def file_generator(input_path, text_vec_dict=None, \
                   V=5, seq_length=5, num_neg=3, batch_size=2, \
                   pre_emb_dim=5):

    count = 0

    while True:
        seqs = []
        pos_doc = []

        neg_docs = []
        for i in range(num_neg):
            neg_docs.append([])

        y = []
        inputs = []

        with open(input_path, 'r') as f:

            for line in f:
                buf = line[:-1].split(',')

                for seq in buf[:seq_length]:
                    seqs.append(text_vec_dict[seq])

                pos_doc.append(text_vec_dict[buf[seq_length]])

                for idx, neg_doc in enumerate(buf[seq_length:seq_length+num_neg]):
                    neg_docs[idx].append(text_vec_dict[neg_doc])

                # sparse_binary_crossentropy
                y.append(0)

                count += 1

                if count % batch_size == 0:

                    inputs = []

                    shape = (batch_size, seq_length, pre_emb_dim)
                    seqs = np.array(seqs, dtype=np.float).reshape(shape)
                    #print('seqs.shape', seqs.shape)
                    inputs.append(seqs)

                    shape = (batch_size, pre_emb_dim)
                    pos_doc = np.array(pos_doc, dtype=np.float).reshape(shape)
                    inputs.append(pos_doc)

                    for i in range(num_neg):
                        neg_doc = np.array(neg_docs[i], dtype=np.float).reshape(shape)
                        inputs.append(neg_doc)

                    y = np.array(y, dtype=np.float)

                    yield inputs, y

                    seqs = []
                    pos_doc = []

                    neg_docs = []
                    for i in range(num_neg):
                        neg_docs.append([])

                    y = []

                    count = 0


def attention(inputs, seq_length):
    """
    Input: inputs (batch_size, seq_length, input_dim)
    Output: (batch_size, seq_length, input_dim)
    """
    a = Permute((2, 1))(inputs)
    a = Dense(seq_length, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output = Multiply()([inputs, a_probs])
    return output


def bn_dense(inputs, output_dim=10, activation='sigmoid'):
    outputs = Dense(units=output_dim)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation)(outputs)

    return outputs


def bilstm_attention(seqs_input,
                     pre_emb_dim=5,
                     seq_length=1):
    """
    seqs_input: batch_size * seq_length * pre-trained_dim
    """

    # num_params: 4 * ((x+y) * y + y), W_xh * X_t + W_hh * h_{t-1} + b_h
    seqs = Bidirectional(LSTM(input_dim=pre_emb_dim,\
                              units=32, return_sequences=True),
                              merge_mode='ave')(seqs_input)

    seqs = attention(seqs, seq_length)
    seqs = Flatten()(seqs)

    #print('seqs.shape', seqs.shape)

    seqs = transforming(seqs)

    return seqs


def transforming(inputs):
    #outputs = bn_dense(inputs, output_dim=128, activation='relu')
    #outputs = bn_dense(outputs, output_dim=64, activation='relu')
    outputs = bn_dense(inputs, output_dim=32, activation='sigmoid')

    return outputs


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


def dssm(seq_length=7,
         pre_emb_dim=5,
         num_neg=4):
    """
    Input:
    Output:
    """

    # transforming of query
    query_input = Input(shape=(seq_length, pre_emb_dim, ))
    query = bilstm_attention(query_input,\
                             pre_emb_dim=pre_emb_dim,\
                             seq_length=seq_length)

    # transforming of pos_doc
    pos_doc_input = Input(shape=(pre_emb_dim, ))
    pos_doc = transforming(pos_doc_input)

    print('pos_doc.shape', pos_doc.shape)

    # transforming of neg_docs
    neg_docs_input = [Input(shape=(pre_emb_dim, )) for _ in range(num_neg)]
    neg_docs = [transforming(neg_doc_input)\
                for neg_doc_input in neg_docs_input]
    print('neg_doc.shape', neg_docs[0].shape)

    # compute softmax probability
    prob = compute_prob(query, [pos_doc] + neg_docs)
    print('prob.shape', prob.shape)

    model = Model(inputs=[query_input, pos_doc_input] + neg_docs_input,\
                  outputs=prob)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')
    return model


def load_dict(dict_path):
    text_vec_dict = {}
    fr = open(dict_path, 'r')

    for line in fr:
        buf = line[:-1].split('\t')
        text_id = buf[0]
        text_vec = list(np.array(buf[1].split(','), dtype=np.float))

        text_vec_dict[text_id] = text_vec

    fr.close()

    return text_vec_dict


def train_model():
    V = 10 # vocabulary size

    seq_length = 3
    num_neg = 2

    pre_emb_dim = 5

    model = dssm(seq_length=seq_length,\
                 pre_emb_dim=pre_emb_dim,\
                 num_neg=num_neg)

    model.summary()

    epochs = 100
    workers = 4
    num_samples = 100
    batch_size = 10
    steps_per_epoch = int(num_samples / batch_size)

    print('Loading text_vec_dict')
    # V * pre_emb_dim (pretrained_embedding_dim)

    text_vec_dict_path = 'data/text_vec.dict'

    text_vec_dict = load_dict(text_vec_dict_path)

    print('Load text_vec_dict Done!')

    input_path = 'data/input.txt'
    model.fit_generator(file_generator(input_path=input_path,
                                       text_vec_dict=text_vec_dict,
                                       V=V,
                                       num_neg=num_neg,
                                       seq_length=seq_length,
                                       batch_size=batch_size,
                                       pre_emb_dim=pre_emb_dim),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        workers=workers,
                        use_multiprocessing=True,
                        verbose=1
                        )

    model.save_weights("models/dssm.model")


if __name__ == '__main__':
    train_model()