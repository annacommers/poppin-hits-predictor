import spotipy
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spotipy.oauth2 import SpotifyClientCredentials

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
    song_data_frame_1 = playlist_data(playlist_id_1)
    song_data_frame_2 = playlist_data(playlist_id_2)
    fig, axs = plt.subplots()
    axs.boxplot([song_data_frame_1[quality], song_data_frame_2[quality]])
    plt.show()

def format_data():
    """
    
    Args: 
        data: A data frame of the data from a playlist.
    Returns:
        matrix: Data in the form for matrix multiplication.
    """
    pass

def find_linear_fit(playlist_1='37i9dQZF1DX2L0iB23Enbq', playlist_2='37i9dQZEVXbMDoHDwVN2tF'):
    """
    Args:
        playlist_1: A Spotify playlist id to analyse how much a song fits with it.
        playlist_2: A control Spotify playlist id
        song:
    Returns:
        How much the song fits with the given playlist_1.
    """
    matrix_1 = get_playlist_data(playlist_1)
    matrix_2 = get_playlist_data(playlist_2)
    both_playlists = np.append(matrix_1, matrix_2, axis=0)
    zeros_matrix = np.zeros((np.shape(both_playlists)[0],np.shape(both_playlists)[0] - np.shape(both_playlists)[1]))
    both_playlists_final = np.append(both_playlists, zeros_matrix, axis=1)
    vector_1 = np.ones((np.shape(matrix_1)[0],1))
    vector_2 = np.zeros((np.shape(matrix_2)[0],1))
    both_vectors = np.append(vector_1, vector_2, axis=0)
    print(both_playlists_final)
    print(both_vectors)
    print(np.shape(both_playlists_final))
    print(np.linalg.inv(both_playlists_final))
find_linear_fit()

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