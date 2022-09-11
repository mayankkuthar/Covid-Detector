import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
import os
from IPython.display import Image, display
import matplotlib.cm as cm1
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="static/cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm1.get_cmap("gnuplot2")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

def ml_solution(xray_name, xray_path):
    TRAIN_PATH = "CovidDataset/Train"
    VAL_PATH = "CovidDataset/Test"

    output = {
        'answer':"",
    }


    #CNN of Model
    '''model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))

    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])'''

    #Training of dataset
    train_datagen = image.ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
        )

    test_dataset = image.ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
        'CovidDataset/Train',
        target_size = (224,224),
        batch_size = 32,
        class_mode = 'binary'
        )

    train_generator.class_indices

    validation_generator = test_dataset.flow_from_directory(
        'CovidDataset/Val',
        target_size = (224,224),
        batch_size = 32,
        class_mode = 'binary'
        )

    validation_generator.class_indices

    '''hist = model.fit(
    train_generator,
    steps_per_epoch = 7,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps = 2
    )

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")'''

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model1 = model_from_json(loaded_model_json)

    # load weights into new model
    model1.load_weights("model.h5")
    print("Loaded model from disk")

    model1.compile(loss=keras.losses.binary_crossentropy, optimizer='rmsprop', metrics=['accuracy'])
    model1.evaluate(train_generator)
    model1.evaluate(validation_generator)

    image1 = image.load_img(xray_path, target_size=(224, 224))
    img = np.array(image1)
    img = img / 255
    img = img.reshape(1,224,224,3)
    label = model1.predict(img)
    print(label[0][0].round(6))
    if(label[0][0].round(0) == 0):
        output['answer'] = "Positive"
    else:
        output['answer'] = "Negative"


    train_generator.class_indices
    y_actual = []
    y_test = []


    for i in os.listdir('./CovidDataset/Val/Normal/'):
        img = image.load_img('./CovidDataset/Val/Normal/'+i, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        p = model1.predict(img)
        y_test.append(p[0,0])
        y_actual.append(1)
        
    for i in os.listdir('./CovidDataset/Val/Covid/'):
        img = image.load_img('./CovidDataset/Val/Covid/'+i, target_size=(224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        p = model1.predict(img)
        y_test.append(p[0,0])
        y_actual.append(0)
        
    y_actual = np.array(y_actual)
    y_test = np.array(y_test)

    cm = confusion_matrix(y_actual, y_test.round(0))
    #print(classification_report(y_actual, y_test.round(4)))

    heat_m =sns.heatmap(cm, cmap='coolwarm', annot=True)
    plt.savefig("static/confmat.jpg")

    # Display

    model_builder = keras.applications.xception.Xception
    img_size = (299, 299)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    last_conv_layer_name = "block14_sepconv2_act"

    # The local path to our target image
    img_path = xray_path

    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # Make model
    model = model_builder(weights="imagenet")

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Print what the top predicted class is
    preds = model.predict(img_array)

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.savefig('./static/heatmap.jpg')

    save_and_display_gradcam(img_path, heatmap)

    return output