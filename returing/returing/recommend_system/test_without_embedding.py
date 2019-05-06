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

    V = 100
    embedding_dim = 8
    input_length = 7

    """
    weights = load_weights()

    embedding_layer = Embedding(input_dim=V,
                       output_dim=embedding_dim,
                       input_length=input_length,
                       trainable=False,
                       weights=weights,
                  name='embedding')
    """

    inputs = Input(shape=(input_length, embedding_dim))

    #y = embedding_layer(inputs) # !!! Remove embedding layer in TEST !!!

    y = Flatten(name='flatten')(inputs)
    y = Dense(output_dim=4, activation='relu', name='dense_1')(y)
    y = Dense(output_dim=1, activation='sigmoid', name='dense_2')(y)

    model = Model(inputs=inputs, outputs=y)

    return model


def main():

    model = build_model()

    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.summary()

    model_weights_path = 'models/with_embedding.weights'


    model.load_weights(model_weights_path, by_name=True)
    #layer = model.get_layer("embedding")
    #weights = layer.get_weights()

    test_batch = 4

    #V = 100
    n_input = 7
    embedding_dim = 8

    inputs = np.random.randn(test_batch, n_input, embedding_dim)
    #print(inputs.shape)

    y_pred = model.predict(inputs)
    print(y_pred)


if __name__ == '__main__':
    main()