# poppin-hits-predictor

This project is for analyzing song data from Spotify songs using principal component analysis. This coade attempts to show similarities between two groups of songs using data provided by spotify.

## Usage

To visualize the two groups of songs you can run the function `run_algorithm` or `run_local_algorithm`. The `run_algorithm` function uses the Spotify API whereas the `run_local_algorithm` function uses data stored locally in CSV files. More information about the use of these functions is detailed bellow.

### Spotify API

To access the Spotify API you will need your own Spotify API client id and client secret. Once these have been aquired, you must export them as enviornment variables to use them with the code. In Ubuntu you can type the following into your terminal:

```bash
export SPOTIPY_CLIENT_SECRET='insert_client_secret'
export SPOTIPY_CLIENT_ID='insert_client_id'
```
You can learn more about aquiring these credentials in the official [Spotipy documentation](https://spotipy.readthedocs.io/en/2.12.0/#client-credentials-flow).

Once you have set the credentials for Spotipy you can run the function `run_algorithm` to see the output of the code. The function will produce a graph of compressed song data plotted on two principle vectors. This graph should help illustrate how much the different playlists overlap. The `run_algorithm` function requires four parameters, two spotify playlist ids and their associated grouping name. A general function call is shown bellow:

```python
run_algorithm("playlist_1_ID", "playlist_2_ID", "group_name_1", "group_name_2")
``` 

### Local Data

If you are unable to aquire spotify credentials it is also possible to test the code using data that has already been downloaded into CSV files. These files are stored in the data folder. Once you have selected the two files of song data you'd like to compare, you can use the `run_local_algorithm` function to visualize the data. The functions takes four parameters two file paths and their associated grouping name. A general function call is shown bellow:

```python
run_local_algorithm("file_name_1", "file_name_2", "group_name_1", "group_name_2")
``` 

### Downloading More Data

If you would like to download your own playlist data you can use the `save_playlist_data` function which takes a file name (excluding the file tag) and the playlist id which contains the songs you'd like to collect data from. An example of a possible function call is shown bellow:
```python
save_playlist_data("file_name","playlist_id")
```

## Authors and Acknowlegment 

This project was written by Luke Nonas-Hunter and Anna Commers with guidance from Steve Matsumoto, for the software design class at Olin College of Engineering.