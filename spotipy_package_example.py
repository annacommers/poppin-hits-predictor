import spotipy
import sys
from spotipy.oauth2 import SpotifyClientCredentials

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

if len(sys.argv) > 1:
    playlist_id = ' '.join(sys.argv[1:])
else:
    playlist_id = '37i9dQZF1DX2L0iB23Enbq'

results = spotify.playlist(playlist_id, fields="tracks")
items = results['tracks']['items']
for entry in items:
    print(entry['track']['name'])