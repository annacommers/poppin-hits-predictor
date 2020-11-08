"""
Tests for Poppin Hits Predictor
"""
import pytest
from pytest import approx
import numpy as np

from poppin_hits_predictor import(
    compress_data,
    pca,
    nearest_neighbor,
    calculate_accuracy,
    run_algorithm,
)

def test_pca():
    matrix = [[2, 4], [3, 1]]
    label = np.array([True, False])
    data = (matrix, np.all(label))
    assert pca(data, 1) == np.all([4, 3])


def test_compress_data():
    matrix = [[2, 4], [3, 1]]
    data = (matrix, [True, False])
    vector = [4, 3]
    compressed_data = [2.5, -2.5]
    label = [True, False]
    #use approx to avoid floating point errors
    assert compress_data(data, vector) == (approx(compressed_data), label)


def test_nearest_neighbor():
    #check that training and testing with the same matrix will have 100%
    #accuracy
    matrix = [[2, 4], [3, 1]]
    labels = np.array([True, False])
    data1 = (matrix, np.all(labels))
    data2 = (matrix, np.all(labels))
    testing_results = np.array([True, True])
    #testing_results = np.full((test_song_labels.shape), True)
    assert nearest_neighbor(data1, data2) == (np.all(testing_results), labels)


# def test_run_algorithm():
#     playlist_1 = '37i9dQZF1DX2L0iB23Enbq'
#     assert run_algorithm() == 
