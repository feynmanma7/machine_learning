#encoding:utf-8
import keras
from keras import Model
from keras.layers import Embedding, Dense, Input, Flatten
import numpy as np
np.random.seed(20170430)
import os


def load_weights():

    weights = []
    V = 100
    embedding_dim = 8
    weights.append(np.random.randn(V, embedding_dim))

    return weights


def build_model():

    weights = load_weights()

    V = 100
    embedding_dim = 8
    input_length = 7

    embedding_layer = Embedding(input_dim=V,
                       output_dim=embedding_dim,
                       input_length=input_length,
                       trainable=False,
                       weights=weights,
                  name='embedding')

    """
    embedding_layer = Embedding(input_dim=V,
                                output_dim=embedding_dim,
                                input_length=input_length,
                                name='embedding')
    """

    inputs = Input(shape=(input_length, ))

    y = embedding_layer(inputs)

    y = Flatten(name='flatten')(y)
    y = Dense(output_dim=4, activation='relu', name='dense_1')(y)
    y = Dense(output_dim=1, activation='sigmoid', name='dense_2')(y)

    model = Model(inputs=inputs, outputs=y)

    return model


class DataGenerator(keras.utils.Sequence):

    def __init__(self, input_path=None, line_offset=None):
        super(DataGenerator, self).__init__()

        self.input_path = input_path
        self.line_offset = line_offset

    def __len__(self):
        total_num = 200
        batch_size = 5
        n_batch = total_num / batch_size
        return int(n_batch)

    def __getitem__(self, idx):
        #input_path = 'data/train.txt'
        fr = open(self.input_path, 'r')
        fr.seek(self.line_offset[idx])

        line_count = 0

        batch_size = 5

        inputs = []
        count = 0
        for line in fr:
            """
            line_count += 1

            if line_count < idx * batch_size or \
                line_count >= (idx + 1) * batch_size:
                continue
            """

            count += 1
            data = np.array(line[:-1].split(','), dtype=np.float32)
            inputs.append(data)

            if count % batch_size == 0:
                inputs = np.array(inputs, dtype=np.float32)
                y = np.random.randn(inputs.shape[0])

                fr.close()
                return inputs, y

        #print(inputs)
        #return inputs, y


def train_generator():

    batch_size = 5
    n_input = 7
    V = 100

    #pid = os.getpid()
    while True:

        fr = open('data/train.txt', 'r')

        inputs = []

        line_count = 0
        for line in fr:
            line_count += 1
            data = np.array(line[:-1].split(','), dtype=np.float32)
            inputs.append(data)

            if line_count % batch_size == 0:
                line_count = 0
                batch_size = np.random.randint(low=3, high=10)
                inputs = np.array(inputs, dtype=np.float32)
                y = np.random.randn(inputs.shape[0])

                yield inputs, y
                inputs = []

        fr.close()

        #inputs = np.random.randint(low=0, high=V, size=batch_size*n_input).reshape((batch_size, n_input))



def main():

    model = build_model()

    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.summary()

    model_weights_path = 'models/with_embedding.weights'

    """
    model.load_weights(model_weights_path)
    layer = model.get_layer("embedding")
    weights = layer.get_weights()
    print(weights[0].shape)
    return
    """

    total_num = 100
    batch_size = 5
    epochs = 2

    input_path = 'data/train.txt'
    line_offset = []
    offset = 0
    for line in open(input_path, 'r'):
        line_offset.append(offset)
        offset += len(line)

    model.fit_generator(
        #generator=train_generator(),
        generator=DataGenerator(input_path=input_path, line_offset=line_offset),
                        epochs=epochs,
                        steps_per_epoch=total_num/batch_size,
                        use_multiprocessing=True,
                        workers=4
                        )




    model.save_weights(model_weights_path)


if __name__ == '__main__':
    main()