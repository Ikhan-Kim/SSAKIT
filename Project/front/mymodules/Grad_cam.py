import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras_preprocessing.image import ImageDataGenerator

#path
CLASS_NAME = class_name
IMAGE_NAME = image_name
MODEL_NAME = model_name
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(BASE_DIR, 'final1/test')
IMAGE_PATH = os.path.join(BASE_DIR, 'final1/test/'+CLASS_NAME +'/' +IMAGE_NAME +'.png')
MODEL_PATH = os.path.join(BASE_DIR, 'test_Pjt3/checkpoint/'+MODEL_NAME +'.h5')

# grad-cam 
def get_img_array(img_path, size):

    image = tf.keras.preprocessing.image.load_img(img_path, target_size=size)

    image_array = tf.keras.preprocessing.image.img_to_array(image)

    image_array = np.expand_dims(array, axis=0)
    return image_array


def make_gradcam_heatmap(
    image, base_model, last_conv_layer_name, classifier_layer_names):

    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(base_model.input, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    for layer_name in classifier_layer_names:
        x = base_model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(image)
        tape.watch(last_conv_layer_output)

        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap



def VGG16_Grad_cam():
    # Define hyperparameter
    INPUT_SIZE = 224
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)

    last_conv_layer_name = 'block5_conv3'
    classifier_layer_names = [
        'block5_pool',
        'flatten',
        'fc1',
        'fc2',
        'predictions',
    ]

    # Load pre-trained model
    base_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet', input_shape=INPUT_SHAPE)

    base_model.trainable = False

    # load model
    new_model = tf.keras.models.load_model(MODEL_PATH)

    image1 = tf.keras.applications.vgg16.preprocess_input(get_img_array(IMAGE_PATH, size=(INPUT_SIZE, INPUT_SIZE)))

    heatmap = make_gradcam_heatmap(
        image1, base_model, last_conv_layer_name, classifier_layer_names,
    )

    # overlay V2
    image_origin = tf.keras.preprocessing.image.load_img(IMAGE_PATH)
    image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)


    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # heatmap 사진을 원본 사이즈에 맞추기
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((imgage_origin.shape[1], imgage_origin.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # 원본 사진과 heatmap사진을 겹치기
    superimposed_img = jet_heatmap * 0.7 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # plt.matshow(superimposed_img)
    # plt.show()
    return superimposed_img


def EFFICIENTNETB0_Grad_cam():
    # Define hyperparameter
    INPUT_SIZE = 224
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)

    last_conv_layer_name = 'top_conv'
    classifier_layer_names = [
        'top_bn',
        'top_activation',
        'avg_pool',
        'top_dropout',
        'predictions',
    ]

    # Load pre-trained model
    base_model = tf.keras.applications.EfficientNetB0(include_top=True, weights='imagenet', input_shape=INPUT_SHAPE)

    base_model.trainable = False

    # load model
    new_model = tf.keras.models.load_model(MODEL_PATH)

    image1 = tf.keras.applications.vgg16.preprocess_input(get_img_array(IMAGE_PATH, size=(INPUT_SIZE, INPUT_SIZE)))

    heatmap = make_gradcam_heatmap(
        image1, base_model, last_conv_layer_name, classifier_layer_names,
    )

    # overlay V2
    image_origin = tf.keras.preprocessing.image.load_img(IMAGE_PATH)
    image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)


    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # heatmap 사진을 원본 사이즈에 맞추기
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((imgage_origin.shape[1], imgage_origin.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # 원본 사진과 heatmap사진을 겹치기
    superimposed_img = jet_heatmap * 0.7 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # plt.matshow(superimposed_img)
    # plt.show()
    return superimposed_img

def RESNET50_Grad_cam():
    # Define hyperparameter
    INPUT_SIZE = 224
    CHANNELS = 3
    INPUT_SHAPE = (INPUT_SIZE, INPUT_SIZE, CHANNELS)

    last_conv_layer_name = 'conv5_block3_3_conv'
    classifier_layer_names = [
        'conv5_block3_3_bn',
        'conv5_block2_out',
        'conv5_block3_3_bn',
        'conv5_block3_out',
        'avg_pool',
        'predictions',
    ]

    # Load pre-trained model
    base_model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', input_shape=INPUT_SHAPE)

    base_model.trainable = False

    # load model
    new_model = tf.keras.models.load_model(MODEL_PATH)

    image1 = tf.keras.applications.vgg16.preprocess_input(get_img_array(IMAGE_PATH, size=(INPUT_SIZE, INPUT_SIZE)))

    heatmap = make_gradcam_heatmap(
        image1, base_model, last_conv_layer_name, classifier_layer_names,
    )

    # overlay V2
    image_origin = tf.keras.preprocessing.image.load_img(IMAGE_PATH)
    image_origin = tf.keras.preprocessing.image.img_to_array(image_origin)


    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # heatmap 사진을 원본 사이즈에 맞추기
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_origin.shape[1], image_origin.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # 원본 사진과 heatmap사진을 겹치기
    superimposed_img = jet_heatmap * 0.7 + image_origin
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # plt.matshow(superimposed_img)
    # plt.show()
    return superimposed_img