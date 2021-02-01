import glob
import numpy as np
import ntpath
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from numpy . random import seed
seed (1234)
from tensorflow import set_random_seed
set_random_seed (4321)

########use here to change hyperparameters and parameters
number_of_epochs = 10
learning_rate = 0.001
img_index_to_predict = 2

train_set_file_path = "dataset/train_set/*.jpg"
test_set_file_path = "dataset/test_set/*.jpg"
train_set_label_file_path = "dataset/train_set_label.txt"
test_set_label_file_path = "dataset/test_set_label.txt"
color_codes_file_path = "dataset/color_codes.txt"
########

x_train, y_train, x_test, y_test = [], [], [], []
labels = {}
color_codes = {}

#load img files and (train/test)_set_label.txt
def load_data(file_path, label_file_path):
    labels = {}
    with open(label_file_path, 'r') as file:
        for line in file:
            line = line.rstrip().split()
            labels[line[0]] = line[1]

    files = glob.glob(file_path)
    x, y = [], []
    for file in files:
        img = image.load_img(file)
        file = ntpath.basename(file)
        # preprocessing
        img = tf.keras.utils.normalize(img, axis=1)
        x.append(img)
        y.append(labels[file])
    return x, y

def load_color_codes():
    with open(color_codes_file_path, 'r') as file:
        for line in file:
            line = line.rstrip().split()
            color_codes[line[0]] = line[1]

def one_hot_encode(data):
    encoded = to_categorical(data)
    return encoded

#plots accuracy
def show_accuracy():
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.clf()

##plots loss
def show_loss():
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.clf()

if __name__ == '__main__':
    print('program started...')

    # load training data and train_set_label.txt
    x_train, y_train = load_data(train_set_file_path, train_set_label_file_path)
    y_train = one_hot_encode(y_train)

    # converting list into numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #creating model
    ###code for 1 layer network
    """
    ###code for 1 layer network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(128, 64, 3)))
    model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))
    """

    ###code for 2 layer network
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(128, 64, 3)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))
    """

    ###code for 3 layer network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(128, 64, 3)))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(11, activation=tf.nn.softmax))

    #compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #fit model
    history = model.fit(x_train, y_train, validation_split=0.20, epochs=number_of_epochs)

    #show history
    show_accuracy()
    show_loss()

    # load test data and test_set_label.txt
    x_test, y_test = load_data(test_set_file_path, test_set_label_file_path)
    y_test = one_hot_encode(y_test)

    # converting list into numpy array
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    #evaluate model
    loss, acc = model.evaluate(x_test, y_test)
    print("Accuracy: {0} % , Loss: {1} %,".format(round(acc * 100, 4), round(loss, 4)))

    print('program finished...')

"""
    #save model, predictions
    model.save('color_predictor')
    new_model = tf.keras.models.load_model('color_predictor')
    predictions = new_model.predict(x_test)

    #loads color codes and prints tshirt color
    load_color_codes()
    i = str(np.argmax(predictions[img_index_to_predict]))
    print(color_codes.get(i))

    plt.imshow(x_test[img_index_to_predict])
    plt.show()
"""

