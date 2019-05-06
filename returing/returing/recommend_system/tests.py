import numpy as np
np.random.seed(20170430)
import time
from keras import Sequential
from keras.layers import Dense

import pickle
import threading
from queue import Queue
import sys

def build_model():

    model = Sequential()

    model.add(Dense(units=10, input_shape=(12, ), activation='sigmoid'))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='sgd', loss='binary_crossentropy')

    return model


def producer(p_q, total_size, n_thread):
    n_epoch = 100
    for i in range(total_size * n_epoch):
        p_q.put(i)

    for _ in range(n_thread * 2):
        p_q.put(total_size+1)

    return


def consumer(p_q, c_q, batch_size, total_size):

    while True:

        i = p_q.get()
        if i == total_size + 1:
            return

        X = np.random.randn(batch_size, 12)
        y = np.random.randint(0, 1, size=batch_size)

        c_q.put((X, y))


def generator(c_q):

    while True:
        (X, y) = c_q.get()
        yield X, y


if __name__ == '__main__':

    p_q = Queue(maxsize=100)
    batch_size = 5
    n_thread = 10

    total_size = 100
    c_q = Queue(maxsize=total_size)

    p = threading.Thread(target=producer, args=(p_q, total_size, n_thread))
    p.start()

    cs = []
    for i in range(n_thread):
        cs.append(threading.Thread(target=consumer, args=(p_q, c_q, batch_size, total_size)))
        cs[i].start()

    model = build_model()


    #batch_size = 5
    workers = 4
    steps_per_epoch = total_size / batch_size

    model.summary()

    model.fit_generator(epochs=100,
                        workers=workers,
                        steps_per_epoch=steps_per_epoch,
                        generator=generator(c_q))

    print('Done!')

    p.join()
    for c in cs:
        c.join()


    sys.exit(0)

