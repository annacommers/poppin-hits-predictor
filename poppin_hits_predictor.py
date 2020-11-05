"""
Poppin Hits Predictor
Anna Commers and Luke Nonas-Hunter

Predicting those poppin hits since 1875
"""

import spotipy
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.neighbors import NearestNeighbors
import math
import helper_functions as helper

def get_playlist_data(spotify, playlist_id, playlist_label):
    """
    Return numerical playlist data and labels for song type.

    Given a playlist id and a boolean value identifying which type of playlist,
    return a matrix of numerical playlist data and an array of boolean values
    which can be used to identify which type of playlist each song came from.
    
    Args:
        playlist_id: The identification string of a single Spotify playlist.
        playlist_label: A boolean value distingishing which playlist type is
        used.
    Returns:
        playlist_data: A matrix with the numerical playlist data of the
        playlist given.
        song_labels: An array of boolean values equal to the playlist_label
        argument, the length of the number of songs in the playlist given.
    """
    song_ids = helper.get_song_ids(spotify, playlist_id)
    song_data = []
    for identification in song_ids:
        data = spotify.audio_features(identification)
        single_song_data = []
        for key in data[0]:
            if (type(data[0][key]) is float) or (type(data[0][key]) is int):
                single_song_data.append(data[0][key])
        song_data.append(single_song_data)
    playlist_data = np.matrix(song_data)
    playlist_data = playlist_data / playlist_data.max(axis=0)
    # if playlist_label:
    #     song_labels = np.ones((1, len(song_ids)), dtype=bool)
    # else:
    #     song_labels = np.zeros((1, len(song_ids)), dtype=bool)
    song_labels = np.full(len(song_ids), playlist_label)
    return (playlist_data, song_labels)


def compress_data(data, principle_vectors):
    """
    Return playlist data compressed with principle eigenvectors.

    Given data containing playlist data and song labels, and a number of
    principle eigenvectors, return a matrix of compressed song data and the
    same song labels contained in the data given.
    
    Args:
        data: A tuple containing a matrix of playlist data and an array of song
        labels.
    Returns:
        compressed_song_data: A matrix with compressed playlist data.
        song_labels: An array of boolean values which is the same as song
        labels in the data given.
    """
    playlist_data, song_labels = data
    mean_centered = playlist_data - np.mean(playlist_data, axis=0)
    vector_transpose = np.transpose(principle_vectors)
    compressed_song_data = np.matmul(mean_centered, vector_transpose)
    return (compressed_song_data, song_labels)


def pca(data, num_principle_vectors):
    """
    Return a number of principle eigenvectors of a matrix.

    Given data containing playlist data and song labels, and the number of
    principle eigenvectors desired, return that number of principle
    eigenvectors from the matrix of playlist data. 
    
    Args:
        data: A tuple containing a matrix of playlist data and an array of song
        labels.
    Returns:
        principle_vectors: An array of principle eigenvectors.
    """
    playlist_data, song_labels = data
    mean_centered = playlist_data - np.mean(playlist_data, axis=0)
    A = (1/math.sqrt(np.shape(mean_centered)[0]-1)) * mean_centered
    covariance_matrix = np.transpose(A) * A
    value, vector = np.linalg.eig(covariance_matrix)
    principle_vectors = np.zeros((num_principle_vectors, vector.shape[0]))
    for index in range(num_principle_vectors):
        index_max = np.argmax(value)
        value[index_max] = -np.inf
        principle_vectors[index] = np.transpose(vector[:,index_max])
    return principle_vectors


def nearest_neighbor(training_data, testing_data):
    """
    Use a nearest neighbor algorithm to find the predicted 
    label of a given set of test data.

    Args:
        training_data: Data used from training the nearest neighbor algorithm.
        testing_data: Data used to test the nearest neighbor algorithm.
    Returns:
        The predicted labels of the testing data
    """
    principle_vectors = pca(training_data, 2)
    compressed_training_data, training_song_labels = compress_data(
        training_data,
        principle_vectors)
    compressed_testing_data, testing_song_labels = compress_data(
        testing_data,
        principle_vectors)

    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(compressed_training_data)
    distances, indecies = neighbors.kneighbors(compressed_testing_data)

    testing_results = np.full((testing_song_labels.shape), False)
    testing_results[indecies]= training_song_labels[indecies]
    return (testing_results, testing_song_labels)

        

def graph_compressed_data(data):
    principle_vectors = pca(data, 2)
    compressed_data, labels = compress_data(data, principle_vectors)
    plt.plot(compressed_data[labels][:,0],compressed_data[labels][:,1], 'bo', label='TikTok Song')
    plt.plot(compressed_data[~labels][:,0],compressed_data[~labels][:,1], 'rx', label='Top Chart Song')
    plt.legend()
    plt.title("Compressed Song Data [TikTok Songs vs Top Chart Songs]")
    plt.xlabel("Eigen Vector 1")
    plt.ylabel("Eigen Vector 2")
    plt.show()
    

def calculate_accuracy(testing_results):
    return percent_accurate


def run_algorithm():
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    playlist_data_1 = get_playlist_data(spotify, '37i9dQZF1DX2L0iB23Enbq', True)
    # Jazz Songs: 37i9dQZF1DXbITWG1ZJKYt
    # Top Charts: 37i9dQZEVXbMDoHDwVN2tF
    playlist_data_2 = get_playlist_data(spotify, '37i9dQZEVXbMDoHDwVN2tF', False)
    song_data = np.concatenate((playlist_data_1[0], playlist_data_2[0]))
    label = np.concatenate((playlist_data_1[1], playlist_data_2[1]))

    choice = np.random.choice(range(song_data.shape[0]), size=(int(song_data.shape[0]/2),), replace=False)    
    sample_1 = np.zeros(song_data.shape[0], dtype=bool)
    sample_1[choice] = True
    sample_2 = ~sample_1
    training_song_data = song_data[sample_1, :]
    testing_song_data = song_data[sample_2, :]
    training_label = label[sample_1]
    testing_label = label[sample_2]
    training_data = (training_song_data, training_label)
    testing_data = (testing_song_data, testing_label)

    results = nearest_neighbor(training_data, testing_data)
    
    count = results[0].shape[0]
    correct = 0
    for index in range(results[0].shape[0]):
        if results[0][index] == results[1][index]:
            correct += 1

    print(correct/count)   

    print(pca((song_data[label], label[label]), 1))

    graph_compressed_data((song_data, label))
run_algorithm()