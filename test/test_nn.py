# TODO: import dependencies and write unit tests below

import numpy as np
import sys
import os 
import pytest

sys.path.append(os.path.abspath("/Users/sophieporak/Documents/GitHub/final-nn"))

from nn import nn, preprocess

""""
again, help from ChatGPT and Isaiah Hazelwoods code

not to self np.isclose gives me an array of boolean, whereas np.allclose returns a single boolean. 
"""

def test_single_forward():
    network = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}],
        lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='mean_square_error'
        )
    A_prev = np.array([[1.0, -1.0]])
    W = np.array([[1.0, -1.0], [-1.0, 1.0]])
    b = np.array([[0.0], [0.0]])
    A, Z = network._single_forward(W, b, A_prev, "relu")

    expected_Z = A_prev @ W.T + b.T
    expected_A = np.maximum(0, expected_Z)

    assert np.allclose(Z, expected_Z)
    assert np.allclose(A, expected_A)

def test_forward():
    network = nn.NeuralNetwork(
        nn_arch=[
            {'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
            {'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}
        ],
        lr=0.01, seed=0, batch_size=1, epochs=1, loss_function='binary_cross_entropy'
    )
    X = np.array([[0.5, -0.5]])
    y_hat, cache = network.forward(X)

    assert isinstance(cache, dict)
    assert "A0" in cache #input layer
    assert "Z1" in cache
    assert "A1" in cache 
    assert "Z2" in cache
    assert "A2" in cache #final output layer activation

    assert y_hat.shape == (1,1)   

def test_single_backprop():
    network = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'}],
        lr=0.01, seed=1, batch_size=1, epochs=1, loss_function='mean_square_error')
    A_prev = np.array([[1.0, -1.0]])
    Z_curr = np.array([[1.0, -1.0]])
    dA = np.array([[1.0, 2.0]])
    W = np.array([[1.0, -1.0], [-1.0, 1.0]])
    b = np.array([[0.0], [0.0]])

    dA_prev, dW, db = network._single_backprop(W, b, Z_curr, A_prev, dA, "relu")

    #I want to know if I change this specific weight, how wil the loss change.
    #so for each weight I need a gradient. 
    assert dW.shape == W.shape
    #same for the bias term. each output neuron has a bias term. 
    assert db.shape == b.shape
    #and I want to know how the previous layer influenced the loss. 
    #need one gradient for each activation per input sample.
    assert dA_prev.shape == A_prev.shape
    

def test_predict():
    network = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=0.01, seed=42, batch_size=1, epochs=1, loss_function='binary_cross_entropy'
    )
    X = np.array([[1.0, 2.0]])
    y_pred = network.predict(X)
    assert y_pred.shape == (1, 1)
    assert (y_pred >= 0).all() and (y_pred <= 1).all()

def test_binary_cross_entropy():
    #empty [] means no layers, this is for testing loss only
    network = nn.NeuralNetwork([], 0.01, 0, 1, 1, 'binary_cross_entropy')
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])
    loss = network._binary_cross_entropy(y, y_hat)
    expected = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    assert np.isclose(loss, expected)

def test_binary_cross_entropy_backprop():
    network = nn.NeuralNetwork([], 0.01, 0, 1, 1, 'binary_cross_entropy')
    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.9], [0.1]])
    dA = network._binary_cross_entropy_backprop(y, y_hat)
    expected = - y / y_hat + (1 - y) / (1 - y_hat)
    assert np.allclose(dA, expected)

def test_mean_squared_error():
    network = nn.NeuralNetwork([], 0.01, 0, 1, 1, 'mean_square_error')
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])
    loss = network._mean_squared_error(y, y_hat)
    expected = np.mean(np.sum((y_hat - y) ** 2, axis=1))
    assert np.isclose(loss, expected)

def test_mean_squared_error_backprop():
    network = nn.NeuralNetwork([], 0.01, 0, 1, 1, 'mean_square_error')
    y = np.array([[1], [0]])
    y_hat = np.array([[0.9], [0.1]])
    dA = network._mean_squared_error_backprop(y, y_hat)
    expected = 2 * (y_hat - y)
    assert np.allclose(dA, expected)

def test_sample_seqs():
    seqs = ["AAA", "AAA", "CCC"]
    labels = [True, True, False]
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    assert sampled_seqs == ["AAA", "AAA", "CCC", "CCC"]
    assert sampled_labels == [True, True, False, False]

def test_one_hot_encode_seqs():
    seqs = ["AAA", "ATT", "CCC"]
    encoded_seqs = preprocess.one_hot_encode_seqs(seqs)
    expected = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
                            ])
    #testing for exact equality
    np.testing.assert_array_equal(encoded_seqs, expected)