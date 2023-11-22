import sklearn.decomposition as sk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def PCA(df, feature_name_list, labels, colors, target_names, plot_name=None):
    """

    Computes and plots principal component analysis of given features
    :param plot_name: string of plot title, if required
    :param df: pandas.DataFrame with features in columns, each row corresponds to one measurement
    :param feature_name_list: list of strings of feature names that should be used in PCA
    :param labels: list of integers corresponding to correct class classification
    :param colors: list of strings of color names corresponding to classes
    :param target_names: list of strings of classes names
    :return: X_pca - matrix of principal components
    """
    feature_list = []
    feature_name = str()
    for feature in feature_name_list:
        feature_name = feature_name + feature + ', '
        feature_list.append(np.stack(df[feature]))

    X = np.concatenate(feature_list, axis=1)

    labels = np.array(labels)

    pca = sk.PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)
    fig, ax = plt.subplots()

    for color, i, target_name in zip(colors, np.arange(len(colors)), target_names):
        ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], color=color, alpha=0.8, lw=2, label=target_name)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    if plot_name is None:
        plt.title('PCA based on ' + feature_name)
    else:
        plt.title(plot_name)
    plt.xlabel('1st principal component')
    plt.ylabel('2nd principal component')
    plt.show()


    return X_pca
