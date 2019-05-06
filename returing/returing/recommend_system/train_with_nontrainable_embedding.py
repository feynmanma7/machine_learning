#encoding:utf-8
from keras import Model
from keras.layers import Embedding, Dense, Input, Flatten
import numpy as np
np.random.seed(20170430)


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


def train_generator():

    batch_size = 5
    n_input = 7
    V = 100

    while True:

        inputs = np.random.randint(low=0, high=V, size=batch_size*n_input).reshape((batch_size, n_input))
        y = np.random.randn(batch_size)

        yield  inputs, y


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
    epochs = 10
    model.fit_generator(generator=train_generator(),
                        epochs=epochs,
                        steps_per_epoch=total_num/batch_size,
                        )




    model.save_weights(model_weights_path)


if __name__ == '__main__':
    main()