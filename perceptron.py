#!/usr/bin/env python3

# perceptron with no hidden layers

import numpy as np
from PIL import Image
import os

# DEFINITIONS
# CONSTANTS
IMG_SIZE = 50
INPUT_SIZE = (IMG_SIZE**2)+1
IMG_AMOUNT = 200
L_RATE = 0.01

# FUNCTIONS
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def update_weights(xi, target, W, learning_rate):
	correct = False
	z = np.dot(xi, W)
	a = sigmoid(z)
	prediction = 1 if a >= 0.5 else 0
	correct = True if prediction == target else False
	error = target - a
	W += learning_rate * error * xi
	return W, correct, prediction, a

def load_and_process(image_PATH):
	if not os.path.exists(image_PATH):
		return None
	try:
		img = Image.open(image_PATH).convert('L')
		img = img.resize((IMG_SIZE, IMG_SIZE))
		pixels = (np.array(img).flatten() / 255.0) - 0.5 
		return np.append(pixels, 1.0)
	except:
		return None

# PATHS
circles_PATH = "media/simple/circles"
squares_PATH = "media/simple/squares"
test_PATH = "media/simple/test"

# prepare training data
X, y = [], []

# circles
for i in range(IMG_AMOUNT):
	vec = load_and_process(f"{circles_PATH}/c{i}.jpg")
	if vec is not None:
		X.append(vec)
		y.append(1)

# squares
for i in range(IMG_AMOUNT):
	vec = load_and_process(f"{squares_PATH}/s{i}.jpg")
	if vec is not None:
		X.append(vec)
		y.append(0)

# weights
W = np.random.randn(INPUT_SIZE) * 0.01

# training loop
for epoch in range(100):
	# shuffle
	combined = list(zip(X, y))
	np.random.shuffle(combined)
	X, y = zip(*combined)

	total = 0
	for xi, target in zip(X, y):
		W, correct, p, a = update_weights(xi, target, W, L_RATE)
		total += 1 if correct else 0
		accuracy = total / len(y) * 100
		print(f"epoch [{epoch:2d}] accuracy: [{accuracy:5.1f}%]")

# test with new data
X1, y1 = [], []
for i in range(10):
	vec = load_and_process(f"{test_PATH}/c{i}.jpg")
	if vec is not None:
		X1.append(vec)
		y1.append(1)

for i in range(10):
	vec = load_and_process(f"{test_PATH}/s{i}.jpg")
	if vec is not None:
		X1.append(vec)
		y1.append(0)

print("testing")
total = 0
options = ["circle", "square"]
for xi, target in zip(X1, y1):
	W, correct, p, a = update_weights(xi, target, W, learning_rate)
	total += 1 if correct else 0
	accuracy = np.trunc(total / len(y1) * 100)
	print(f"prediction: [{options[p]}] correct: [{correct}] score: [{accuracy}%] activation: [{a}]")





