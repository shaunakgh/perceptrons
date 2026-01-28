#!/usr/bin/env python3

# multilayer perceptron (sigmoid neurons)

import numpy as np
import idx2numpy
from PIL import Image
import os

IMG_SIZE = 28
INPUT_SIZE = (IMG_SIZE**2)+1
OUTPUT_SIZE = 10
DATA_PATH = "media/complex"
LAYER_SIZE = [INPUT_SIZE, 15, OUTPUT_SIZE]

# datasets from MNIST
# http://yann.lecun.com/exdb/mnist/

# extract and format image and label data
X = idx2numpy.convert_from_file(f'{DATA_PATH}/train/images.idx3-ubyte')
y = idx2numpy.convert_from_file(f'{DATA_PATH}/train/labels.idx1-ubyte')

X = X.astype(np.float32) / 255.0
X = X.reshape(len(X), -1)
X = X[..., np.newaxis]

# find desired network output
def find_target_output(label):
	target_output_layer = np.zeros(OUTPUT_SIZE)
	target_output_layer[label] = 1
	return target_output_layer

# debug
print(f"shape: [{X[0].shape}] datatype: [{X.dtype}] range: [{X.min()}], [{X.max()}]")

weights = [np.random.randn(y, x) for x, y in zip(LAYER_SIZE[:-1], LAYER_SIZE[1:])]
biases = [np.random.randn(y, 1) for y in LAYER_SIZE[1:]]

def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def cost(y_x, label): return np.linalg.norm(y_x-find_target_output(label), ord=2)
def feedforward(a):
	for w, b in zip(weights, biases):
		a = sigmoid(np.dot(w, a)+b)
	return a

a = [0.01, 0.035, 0, 0.87, 0.06, 0.07, 0.024, 0.07, 0.03, 0.08]
print(cost(a, 3))

# stochastic gradient descent
# def SGD(X, epochs, )

# items[:1]     # → ['a']               ← just first
# items[1:]     # → ['b','c','d','e']   ← skip first / tail
# items[:-1]    # → ['a','b','c','d']   ← everything except last / drop last
# items[1:-1]   # → ['b','c','d']       ← remove first AND last
# items[:]      # → whole new copy      ← important trick!