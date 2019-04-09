from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras import Model
import numpy as np
np.random.seed(20170430)

from scipy.spatial.distance import cosine


def predict_by_resnet50():

    model = ResNet50(weights='imagenet')

    img_path = 'elephant_1.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    #print('Predicted:', decode_predictions(preds, top=3)[0])


def load_model():

    """
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc1000').output)
    """

    """
    #base_model = VGG19(weights='imagenet', include_top=False)
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc1').output)
    """


    #base_model = VGG16(weights='imagenet', include_top=False)
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)


    return model


def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def extract_img_feature(model, img_path = 'elephant_1.jpg'):
    img = preprocess_img(img_path)

    score = model.predict(img)

    #print(score.shape)

    score = score.reshape(-1, 1)
    #print(score.shape)
    return score


if __name__ == '__main__':

    print('Start loading model')
    model = load_model()
    print('Load model Done!')



    """
    ele_1 = extract_img_feature(model, img_path='elephant_1.jpg')
    ele_2 = extract_img_feature(model, img_path='elephant_2.jpg')

    tig_1 = extract_img_feature(model, img_path='tiger_1.jpg')
    tig_2 = extract_img_feature(model, img_path='tiger_2.jpg')

    clo_1 = extract_img_feature(model, img_path='cloth_1.jpg')
    clo_2 = extract_img_feature(model, img_path='cloth_2.jpg')

    print('\n\n')
    print('ele_1 vs ele_2 %.4f' % cosine(ele_1, ele_2))
    print('ele_1 vs tig_1 %.4f' % cosine(ele_1, tig_1))
    print('ele_1 vs tig_2 %.4f' % cosine(ele_1, tig_2))
    print('ele_1 vs clo_1 %.4f' % cosine(ele_1, clo_1))
    print('ele_1 vs clo_2 %.4f' % cosine(ele_1, clo_2))
    print('\n\n')

    print('ele_2 vs tig_1 %.4f' % cosine(ele_2, tig_1))
    print('ele_2 vs tig_2 %.4f' % cosine(ele_2, tig_2))
    print('ele_2 vs clo_1 %.4f' % cosine(ele_2, clo_1))
    print('ele_2 vs clo_2 %.4f' % cosine(ele_2, clo_2))
    print('\n\n')

    print('tig_1 vs tig_2 %.4f' % cosine(tig_1, tig_2))
    print('tig_1 vs clo_1 %.4f' % cosine(tig_1, clo_1))
    print('tig_1 vs clo_2 %.4f' % cosine(tig_1, clo_2))
    print('\n\n')

    print('tig_2 vs clo_1 %.4f' % cosine(tig_2, clo_1))
    print('tig_2 vs clo_2 %.4f' % cosine(tig_2, clo_2))
    print('\n\n')

    print('clo_1 vs clo_2 %.4f' % cosine(clo_1, clo_2))
    """

    eye_1 = extract_img_feature(model, img_path='eye_1.jpg')
    eye_2 = extract_img_feature(model, img_path='eye_2.jpg')

    boy = extract_img_feature(model, img_path='boy.jpg')
    girl = extract_img_feature(model, img_path='girl.jpg')

    print('\n\n')
    print('eye_1 vs eye_2 %.4f' % cosine(eye_1, eye_2))
    print('eye_1 vs boy %.4f' % cosine(eye_1, boy))
    print('eye_1 vs girl %.4f' % cosine(eye_1, girl))
    print('\n\n')

    print('eye_2 vs boy %.4f' % cosine(eye_2, boy))
    print('eye_2 vs girl %.4f' % cosine(eye_2, girl))
    print('\n\n')

    print('boy vs girl %.4f' % cosine(boy, girl))




