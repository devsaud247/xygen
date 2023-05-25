import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier


def load_X_encoded():
    return np.load("models/X_encoded_compressed.npy")

def load_kmeans():
    return joblib.load("models/kmeans_model.pkl")

def load_deep_model():
    return load_model("models/encoder_model.h5")

def model_initialize():
    print("Models loading...")

    ###
    model, kmeans, X_encoded = (
        load_deep_model(),
        load_kmeans(),
        load_X_encoded(),
    )

    ###
    train_files = list(pd.read_csv("scripts/train_file.csv")["filepath"])
    test_files = list(pd.read_csv("scripts/test_file.csv")["filepath"])

    ###
    lisp = train_files
    lisp.extend(test_files)

    ###
    clusters_features = []
    cluster_files = []
    for i in [0, 1, 2, 3, 4, 5]:
        i_cluster = []
        i_labels = []
        for iter, j in enumerate(kmeans.labels_):
            if j == i:
                i_cluster.append(X_encoded[iter])
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
    
    knn = KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree',n_jobs=-1)
    knn.fit(np.array(data),np.array(labels))

    return model, kmeans, X_encoded, knn, data, files


