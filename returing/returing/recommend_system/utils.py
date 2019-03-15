from returing.recommend_system.train_model import build_model
import numpy as np
np.random.seed(20170430)


def generate_data():

    input_path = 'input_2.txt'
    num_samples = 6000
    V = 10
    window_size = 3

    fw = open(input_path, 'w')

    for _ in range(num_samples):
        x_y = np.random.randint(0, V, size=window_size*2 + 1)
        #print(','.join(list(x_y)) + '\n')
        fw.write(np.array2string(x_y, separator=',')[1:-1]+'\n')
    fw.close()
    print('Write Done!')


def print_weights():
    model = build_model()
    model.summary()

    model_weights_path = 'w2v.model'
    model.load_weights(model_weights_path)

    embedding_layer = model.layers[0]
    embedding_weights = embedding_layer.get_weights()

    new_model = build_model()
    new_model_weights_path = 'w2v.model.new'
    new_model.load_weights(new_model_weights_path)

    new_embedding_layer = new_model.layers[0]
    new_embedding_weights = new_embedding_layer.get_weights()

    for w, w_2 in zip(embedding_weights[0], new_embedding_weights[0]):
        print(w, w_2)

if __name__ == '__main__':
    print_weights()
    #generate_data()