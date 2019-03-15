import keras.backend as K
from keras import Sequential
from keras.layers import Embedding, Dense, Lambda, Reshape
from keras.utils import np_utils
import numpy as np
np.random.seed(20170430)


def file_generator(input_path, V=10, window_size=3, batch_size=2):

    count = 0

    while True:
        x = []
        y = []

        with open(input_path, 'r') as f:

            for line in f:
                buf = line.split(',')
                x.append(buf[:-1])

                y_ = buf[-1]
                y_ = np_utils.to_categorical(y_, V)
                y.append(y_)

                count += 1

                if count % batch_size == 0:
                    yield np.array(x, dtype=np.float), np.array(y, dtype=np.float)

                    x = []
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


def build_model(V=10,
                embedding_dim=5,
                window_size=3):
    model = Sequential()
    model.add(Embedding(input_dim=V,
                        output_dim=embedding_dim,
                        embeddings_initializer='glorot_uniform',
                        input_length=window_size * 2))

    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)))
    # model.add(Reshape(embedding_dim, ))
    model.add(Dense(V, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    return model

def train_model():
    V = 10 # vocabulary size
    embedding_dim = 5 # embedding_dim
    window_size = 3 # context size = window_size * 2

    model = build_model(V=V,
                        embedding_dim=embedding_dim,
                        window_size=window_size)

    model.summary()

    epochs = 1000
    workers = 4
    num_samples = 100
    batch_size = 10
    steps_per_epoch = int(num_samples / batch_size)

    input_path = 'input.txt'
    model.fit_generator(file_generator(input_path=input_path,
                                       V=V,
                                       window_size=window_size,
                                       batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        workers=workers,
                        use_multiprocessing=True,
                        verbose=1
                        )

    model.save_weights("w2v.model")


if __name__ == '__main__':
    train_model()