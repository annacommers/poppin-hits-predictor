import spotipy
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spotipy.oauth2 import SpotifyClientCredentials
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
import math

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

def get_song_ids(playlist_id='37i9dQZF1DX2L0iB23Enbq'):
    """
    Returns the identification string of each song in a playlist.
    
    Args:
        playlist_id: The identification string of a single Spotify playlist.
    Returns:
        song_ids: A list of song identification strings.
    """
    song_ids = []
    results = spotify.playlist(playlist_id, fields="tracks")
    items = results['tracks']['items']
    for entry in items:
        song_ids.append(entry['track']['id'])
    return song_ids

def get_playlist_data(playlist_id='37i9dQZF1DX2L0iB23Enbq'):
    """
    Get numerical data on all songs in a given playlist using spotipy python
    library and spotify api.
    
    Args:
        playlist_id: Playlist ID from spotify.
    Returns:
        Returns a numpy matrix containing all numerical data from the given songs.
            Data is ordered in the same order it was recieved.
    """
    song_ids = get_song_ids(playlist_id)
    song_data = []
    for identification in song_ids:
        data = spotify.audio_features(identification)
        single_song_data = []
        for key in data[0]:
            
            if (type(data[0][key]) is float) or (type(data[0][key]) is int):
                single_song_data.append(data[0][key])
        song_data.append(single_song_data)
    return np.matrix(song_data)


def plot_two_playlists(playlist_id_1='37i9dQZF1DX2L0iB23Enbq', playlist_id_2='37i9dQZEVXbMDoHDwVN2tF', quality='danceability'):
    song_data_frame_1 = get_playlist_data(playlist_id_1)
    song_data_frame_2 = get_playlist_data(playlist_id_2)
    fig, axs = plt.subplots()
    axs.boxplot([song_data_frame_1[quality], song_data_frame_2[quality]])
    plt.show()


def get_vector(playlist_1='37i9dQZF1DX2L0iB23Enbq', playlist_2='37i9dQZEVXbMDoHDwVN2tF'):
    vector_1 = np.ones((len(get_song_ids(playlist_1)),1))
    vector_2 = np.zeros((len(get_song_ids(playlist_2)),1))
    both_vectors = np.append(vector_1, vector_2, axis=0)
    return both_vectors


def get_matrix(playlist_1='37i9dQZF1DX2L0iB23Enbq', playlist_2='37i9dQZEVXbMDoHDwVN2tF'):
    """
    
    Args: 
        data: A data frame of the data from a playlist.
    Returns:
        matrix: Data in the form for matrix multiplication.
    """
    matrix_1 = get_playlist_data(playlist_1)
    matrix_2 = get_playlist_data(playlist_2)
    both_playlists = np.append(matrix_1, matrix_2, axis=0)
    return both_playlists


def find_linear_fit(playlist_1='37i9dQZF1DX2L0iB23Enbq',
                    playlist_2='37i9dQZEVXbMDoHDwVN2tF'):
    """
    Calculate linear regression of a songs likelyness to
    be a tiktok song using SVD.
    Args:
        playlist_1: A Spotify playlist id to analyse how
        much a song fits with it.
        playlist_2: A control Spotify playlist id
        song:
    Returns:
        How much the song fits with the given playlist_1.
    """

    # Using SVD we can find the approximate inverse of the given matrix
    # With this inverse we calculate the approximate weights
    # to get the best results.
    matrix = get_matrix(playlist_1, playlist_2)
    vector = get_vector(playlist_1, playlist_2)
    transpose = np.transpose(matrix)
    # economy representatation of SVD
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    S = np.multiply(np.identity(len(S)), np.outer(np.ones(len(S)), S))
    S_inverse = np.linalg.inv(S)
    U_transpose = np.transpose(U)
    V_transpose = np.transpose(V)
    SVD_inverse = np.matmul(V_transpose, np.matmul(S_inverse, U_transpose))
    weights = np.matmul(SVD_inverse, vector)
    output = np.matmul(matrix, weights)
    output_rounded = np.rint(output)
    
    # Checks correctness of linear regression
    print(output_rounded == vector)
    count_correct = 0
    for answer in (output_rounded == vector):
        if answer[0]:
            count_correct += 1
    print(count_correct)

    return weights
#find_linear_fit()

def get_mean_centered_data()

def pca(playlist_1='37i9dQZF1DX2L0iB23Enbq',
        playlist_2='37i9dQZEVXbMDoHDwVN2tF',
        num_principle_vectors=1):
    matrix = get_matrix(playlist_1, playlist_2)
    vector = get_vector(playlist_1, playlist_2)
    transpose = np.transpose(matrix)
    mean_centered = matrix - np.mean(matrix, axis=0)
    A = (1/math.sqrt(np.shape(matrix)[0]-1)) * mean_centered
    covariance_matrix = np.transpose(A) * A
    value, vector = np.linalg.eig(covariance_matrix)
    
    principle_vectors = np.zeros((num_principle_vectors, vector.shape[0]))
    for index in range(num_principle_vectors):
        index_max = np.argmax(value)
        value[index_max] = -np.inf
        principle_vectors[index] = np.transpose(vector[:,index_max])
    compressed = np.matmul(mean_centered, np.transpose(principle_vectors))
    return (compressed, principle_vectors)
pca()

def nearest_neighbor(pca_output, testing_data):
    compressed_data, principle_vectors = pca_output
    training_data =  
    testing_data = 
    compressed_testing_data = np.matmul(testing_data,principle_vectors)
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(compressed_training_data)
    neighbors()

def predict_song_category(train_playlist_1='37i9dQZF1DX2L0iB23Enbq',
                          train_playlist_2='37i9dQZEVXbMDoHDwVN2tF',
                          test_playlist_1='1tPsUcoJohUBmM2RaSk1WH',
                          test_playlist_2='6UeSakyzhiEt4NB3UAd6NQ'):
    pass


def plot_eigen_songs(train_playlist_1='37i9dQZF1DX2L0iB23Enbq',
                     train_playlist_2='37i9dQZEVXbMDoHDwVN2tF',
                     test_playlist_1='1tPsUcoJohUBmM2RaSk1WH',
                     test_playlist_2='6UeSakyzhiEt4NB3UAd6NQ'):
    train_vector = get_vector(train_playlist_1, train_playlist_2)
    test_vector = get_vector(test_playlist_1, test_playlist_2)

    

def predict_popularity(parameter_vector, songs):
    """
    Using given vector, predict songs likelyness to be a popular tik tok song
    using a linear fit model.

    Args:
        parameter_vector: The parameter vector found using find_linear_fit().
        songs: A matrix of data from different songs given by playlist_data() or song_data()
    Returns:
        A list of numbers from 0-1 which describes the likeliness of a song
        becoming popular on tik tok.
    """
    pass

def song_data():
    """

    Args:
        song_id: id for a single song.
    Returns:
        A data frame of data for the song.
    """
    pass


# def final(playlist1, lable1, playlist2, lable2, num_vectors):
#     data1 = get(playlist1, lable1)
#     data2 = get(playlist2, lable2)
#     pv = get_pv(data1, num_pv)
#     output1 = compress(data1, pv)
#     pv = get_pv(data2, num_pv)
#     output2 = compress(data2, pv)
#     get_accuracy = nn(output1, output2)
#     graph(output1, output2)

