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
    matrix = [[1, 2], [3, 4]]
    label = np.array([True, False])
    data = (matrix, np.all(label))
    assert np.allclose(pca(data, 1), np.array([np.sqrt(2)/2, np.sqrt(2)/2]))


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
    data1 = (matrix, labels)
    data2 = (matrix, labels)
    testing_results = np.array([True, False])
    #testing_results = np.full((test_song_labels.shape), True)
    assert np.all(nearest_neighbor(data1, data2)[0] == testing_results)
    assert np.all(nearest_neighbor(data1, data2)[1] == labels)

