"""
Poppin Hits Predictor Helper Functions
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


def get_song_ids(spotify, playlist_id='37i9dQZF1DX2L0iB23Enbq'):
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