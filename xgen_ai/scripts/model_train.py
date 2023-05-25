# For commands
import os
from os.path import exists

import time
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings("ignore")
# For array manipulation
import numpy as np
import pandas as pd
import pandas.util.testing as tm

# For visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
from sklearn.manifold import TSNE
import imageio as io

# For model performance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

# For model training"
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, Activation, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, Adagrad, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import joblib
from matplotlib.colors import ListedColormap
from keras.utils.np_utils import to_categorical

from model import encoder_decoder_model


file_path = os.listdir("../images")

absolute_path = os.path.dirname(__file__)


def dataset_formation():
    train_files, test_files = train_test_split(file_path, test_size=0.15)
    print(len(train_files))
    print(len(test_files))

    train_files = pd.DataFrame(train_files, columns=["filepath"])
    test_files = pd.DataFrame(test_files, columns=["filepath"])
    train_files.to_csv("train_file.csv")
    test_files.to_csv("test_file.csv")


def image2array(file_array):

    """
    Reading and Converting images into numpy array by taking path of images.
    Arguments:
    file_array - (list) - list of file(path) names
    Returns:
    A numpy array of images. (np.ndarray)
    """

    image_array = []
    for path in file_array:
        try:
            img = cv2.imread("../images/" + path)
            print(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            image_array.append(np.array(img))
        except:
            pass
    image_array = np.array(image_array)
    image_array = image_array.reshape(image_array.shape[0], 224, 224, 3)
    image_array = image_array.astype("float32")
    image_array /= 255
    return np.array(image_array)


def cnn_model():
    optimizer = Adam(learning_rate=0.001)
    model = encoder_decoder_model()
    model.compile(optimizer=optimizer, loss="mse")
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", vebose=1, patience=6, min_delta=0.0001
    )
    checkpoint = ModelCheckpoint(
        "../models/encoder_model.h5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    model.fit(
        train_data,
        train_data,
        epochs=35,
        batch_size=16,
        validation_data=(test_data, test_data),
        callbacks=[early_stopping, checkpoint],
    )
    return model


def feature_extraction(model, data, layer=4):

    """
    Creating a function to run the initial layers of the encoder model. (to get feature extraction from any layer of the model)
    Arguments:
    model - (Auto encoder model) - Trained model
    data - (np.ndarray) - list of images to get feature extraction from trained model
    layer - (int) - from which layer to take the features(by default = 4)
    Returns:
    pooled_array - (np.ndarray) - array of extracted features of given images
    """

    encoded = K.function([model.layers[0].input], [model.layers[layer].output])
    encoded_array = encoded([data])[0]

    return encoded_array


def get_batches(data, batch_size=1000):

    """
    Taking batch of images for extraction of images.
    Arguments:
    data - (np.ndarray or list) - list of image array to get extracted features.
    batch_size - (int) - Number of images per each batch
    Returns:
    list - extracted features of each images
    """

    if len(data) < batch_size:
        return [data]
    n_batches = len(data) // batch_size

    # If batches fit exactly into the size of df.
    if len(data) % batch_size == 0:
        return [data[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]

    # If there is a remainder.
    else:
        return [
            data[i * batch_size : min((i + 1) * batch_size, len(data))]
            for i in range(n_batches + 1)
        ]


def plot_(
    x, y1, y2, row, col, ind, title, xlabel, ylabel, label, isimage=False, color="b"
):
    plt.subplot(row, col, ind)
    if isimage:
        plt.imshow(x)
        plt.title(title)
        plt.axis("off")
    else:
        plt.plot(y1, label=label, color="g")
        plt.scatter(x, y1, color="g")
        if y2 != "":
            plt.plot(y2, color=color, label="validation")
            plt.scatter(x, y2, color=color)
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)


def results_(query, result):
    def read(img):
        image = cv2.imread("../images/" + img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    plt.figure(figsize=(10, 5))
    if type(query) != type(30):
        plot_(query, "", "", 1, 1, 1, "Query Image", "", "", "", True)
    else:
        plot_(
            read(files[query]),
            "",
            "",
            1,
            1,
            1,
            "Query Image " + files[query],
            "",
            "",
            "",
            True,
        )
    plt.show()
    plt.figure(figsize=(20, 5))
    for iter, i in enumerate(result):
        plot_(
            read(files[i]), "", "", 1, len(result), iter + 1, files[i], "", "", "", True
        )
    plt.show()

def load_deep_model():
    return load_model("../models/encoder_model.h5")

if __name__ == "__main__":

    if not (exists("train_file.csv") and exists("test_file.csv")):
        dataset_formation()

    train_files = list(pd.read_csv("train_file.csv")["filepath"])
    test_files = list(pd.read_csv("test_file.csv")["filepath"])

    train_data = image2array(train_files)
    test_data = image2array(test_files)


    model = load_deep_model() if exists("../models/encoder_model.h5") else cnn_model()

    d = np.concatenate([train_data, test_data], axis=0)
    d.shape

    X_encoded = []
    i = 0
    # Iterate through the full training set.
    for batch in get_batches(d, batch_size=300):
        i += 1
        # This line runs our pooling function on the model for each batch.
        X_encoded.append(feature_extraction(model, batch, 12))

    X_encoded = np.concatenate(X_encoded)

    lisp = train_files
    lisp.extend(test_files)
    print(len(lisp))

    X_encoded_reshape = X_encoded.reshape(
        X_encoded.shape[0], X_encoded.shape[1] * X_encoded.shape[2] * X_encoded.shape[3]
    )
    print("Encoded shape:", X_encoded_reshape.shape)
    np.save("../models/X_encoded_compressed.npy", X_encoded_reshape)

    kmeans = KMeans(n_clusters=6, random_state=0).fit(X_encoded_reshape)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    kmeans_file = "../models/kmeans_model.pkl"
    joblib.dump(kmeans, kmeans_file)

    clusters_features = []
    cluster_files = []
    for i in [0, 1, 2, 3, 4, 5]:
        i_cluster = []
        i_labels = []
        for iter, j in enumerate(kmeans.labels_):
            if j == i:
                i_cluster.append(X_encoded_reshape[iter])
                i_labels.append(lisp[iter])
        i_cluster = np.array(i_cluster)
        clusters_features.append(i_cluster)
        cluster_files.append(i_labels)

    labels = []
    data = []
    files = []
    for iter, i in enumerate(clusters_features):
        data.extend(i)
        labels.extend([iter for i in range(i.shape[0])])
        files.extend(cluster_files[iter])
    print(np.array(labels).shape)
    print(np.array(data).shape)
    print(np.array(files).shape)

    knn = KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree", n_jobs=-1)
    knn.fit(np.array(data), np.array(labels))

    transform = TSNE
    trans = transform(n_components=2)
    values = trans.fit_transform(X_encoded_reshape)

    num = 19  # datapoint
    res = knn.kneighbors(data[num].reshape(1, -1), return_distance=True, n_neighbors=10)
    results_(num, list(res[1][0])[1:])
