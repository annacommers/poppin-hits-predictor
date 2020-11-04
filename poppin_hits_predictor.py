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


def get_playlist_data(playlist_id, playlist_label):
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
    song_ids = helpers.get_song_ids(playlist_id)
    for identification in song_ids:
        data = spotify.audio_features(identification)
        single_song_data = []
        for key in data[0]:
            if (type(data[0][key]) is float) or (type(data[0][key]) is int):
                single_song_data.append(data[0][key])
        song_data.append(single_song_data)
    playlist_data = np.matrix(song_data)
    # if playlist_label:
    #     song_labels = np.ones((1, len(song_ids)), dtype=bool)
    # else:
    #     song_labels = np.zeros((1, len(song_ids)), dtype=bool)
    song_labels = np.full((1, len(song_ids)), playlist_label)
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
    A = (1/math.sqrt(np.shape(matrix)[0]-1)) * mean_centered
    covariance_matrix = np.transpose(A) * A
    value, vector = np.linalg.eig(covariance_matrix)
    principle_vectors = np.zeros((num_principle_vectors, vector.shape[0]))
    for index in range(num_principle_vectors):
        index_max = np.argmax(value)
        value[index_max] = -np.inf
        principle_vectors[index] = np.transpose(vector[:,index_max])
    return principle_vectors


def nearest_neighbor(training_data, testing_data):
    training_song_data, training_song_labels = training_data
    testing_song_data, testing_song_labels = testing_data

def graph(trainging_data, testing_data):
    training_song_data, training_song_labels = training_data
    testing_song_data, testing_song_labels = testing_data
    # plot (trainging_song_data "red")
    # plot (testing_song_data "blue")

def calculate_accuracy(testing_results, testing_data):
    return percent_accurate


# ITS THE FINAL FUNCTION
# def final(playlist1, lable1, playlist2, lable2, num_vectors):
#     data1 = get(playlist1, lable1)
#     data2 = get(playlist2, lable2)
#     pv = get_pv(data1, num_pv)
#     output1 = compress(data1, pv)
#     pv = get_pv(data2, num_pv)
#     output2 = compress(data2, pv)
#     get_accuracy = nn(output1, output2)
#     graph(output1, output2)
