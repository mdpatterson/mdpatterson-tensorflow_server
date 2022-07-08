import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import subprocess
import tempfile
import random
import requests
import json
import threading


def download_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))
    return train_labels, train_images, test_images, test_labels, class_names


def train_and_eval(train_labels, train_images, test_images, test_labels):
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3,
                            strides=2, activation='relu', name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, name='Dense')
    ])
    model.summary()

    testing = False
    epochs = 5

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))
    return model


def save_the_model(model):
    MODEL_DIR = tempfile.gettempdir()
    MODEL_DIR = '/home/mdp/AWS_DevOps_online/blog_projects/blog6/tmp'
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print('\nSaved model:')
    return export_path, MODEL_DIR


def run_TF_serving(MODEL_DIR):
    os.environ["MODEL_DIR"] = MODEL_DIR
    print('MODEL_DIR= ', MODEL_DIR)
    bashCommand = 'nohup tensorflow_model_server --rest_api_port=8501 --model_name=fashion_model --model_base_path=' + MODEL_DIR + ' >server.log 2>&1'
    #bashCommand = 'nohup tensorflow_model_server --rest_api_port=8501 --model_name=fashion_model --model_base_path=/tmp/ > server.log 2>&1 &'
    print('bashCommand=', bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()




def make_preds():
    rando = random.randint(0, len(test_images) - 1)
    data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
    print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    print('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
        class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]))
    return


if __name__ == '__main__':
    print('TensorFlow version: {}'.format(tf.__version__))
    train_labels, train_images, test_images, test_labels, class_names = download_data()
    model = train_and_eval(train_labels, train_images, test_images, test_labels)
    export_path, MODEL_DIR = save_the_model(model)
    print('export_path=',export_path, 'MODEL_DIR=', MODEL_DIR)
    # try:
    # creating thread
    # t1 = threading.Thread(target=run_TF_serving, args=MODEL_DIR)
    # t2 = threading.Thread(target=make_preds)
    # starting thread 1 & 2
    #t1.start()
    #t2.start()
    # wait until thread 1 & 2 are completely executed
    #t1.join()
    #t2.join()
    # both threads completely executed
    #print("Done!")
    # except:
    #    print('Something is not right')
