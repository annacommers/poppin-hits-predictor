import spotipy
import sys
import pandas as pd
import matplotlib.pyplot as plt
from spotipy.oauth2 import SpotifyClientCredentials

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

# if len(sys.argv) > 1:
#     playlist_id = ' '.join(sys.argv[1:])
# else:
#     playlist_id = '37i9dQZF1DX2L0iB23Enbq'
# song_ids = []
# results = spotify.playlist(playlist_id, fields="tracks")
# items = results['tracks']['items']
# for entry in items:
#     song_ids.append(entry['track']['id'])
# print(spotify.audio_features(song_ids[0]))
#for id in song_ids:

def get_song_ids(playlist_id='37i9dQZF1DX2L0iB23Enbq'):
    song_ids = []
    results = spotify.playlist(playlist_id, fields="tracks")
    items = results['tracks']['items']
    for entry in items:
        song_ids.append(entry['track']['id'])
    return song_ids

def playlist_data(playlist_id='37i9dQZF1DX2L0iB23Enbq'):
    song_ids = get_song_ids(playlist_id)
    song_data = []
    for id in song_ids:
        song_data += (spotify.audio_features(id))
    return pd.DataFrame(song_data)


def plot_two_playlists(playlist_id_1='37i9dQZF1DX2L0iB23Enbq', playlist_id_2='37i9dQZEVXbMDoHDwVN2tF', quality='danceability'):
    song_data_frame_1 = playlist_data(playlist_id_1)
    song_data_frame_2 = playlist_data(playlist_id_2)
    plt.plot(song_data_frame_1[quality], 'o', song_data_frame_2[quality], 'x',)
plot_two_playlists()