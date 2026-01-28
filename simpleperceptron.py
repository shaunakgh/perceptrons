#!/usr/bin/env python3

# perceptron with no hidden layers

import numpy as np
from PIL import Image
import os

IMG_SIZE = 50
INPUT_SIZE = (IMG_SIZE**2)+1

# image set
circles_PATH = "media/simple/circles"
squares_PATH = "media/simple/squares"
test_PATH = "media/simple/test"

# load, resize
def load_and_process(image_PATH):
	if not os.path.exists(image_PATH):
		return None
	try:
		img = Image.open(image_PATH).convert('L')
		img = img.resize((IMG_SIZE, IMG_SIZE))
		pixels = np.array(img).flatten() / 255.0
		return np.append(pixels, 1.0)
	except:
		return None

# prepare training data
X, y = [], []

# circles
for i in range(6):
	vec = load_and_process(f"{circles_PATH}/c{i}.jpg")
	if vec is not None:
		X.append(vec)
		y.append(1)

# squares
for i in range(6):
	vec = load_and_process(f"{squares_PATH}/s{i}.jpg")
	if vec is not None:
		X.append(vec)
		y.append(0)

# weights
W = np.random.randn(INPUT_SIZE) * 0.01
learning_rate = 0.1

# training loop
for epoch in range(30):
	correct = 0
	for xi, target in zip(X, y):
		z = np.dot(xi, W)
		prediction = 1 if z > 0 else 0
		error = target - prediction
		W += learning_rate * error * xi
        
		if prediction == target:
			correct += 1

		accuracy = correct / len(y) * 100
		print(f"epoch [{epoch:2d}] accuracy: [{accuracy:5.1f}%]")

# test with new data
X1, y1 = [], []
for i in range(6):
	vec = load_and_process(f"{test_PATH}/c{i}.jpg")
	if vec is not None:
		X1.append(vec)
		y1.append(1)

for i in range(6):
	vec = load_and_process(f"{test_PATH}/s{i}.jpg")
	if vec is not None:
		X1.append(vec)
		y1.append(0)

for xi, target in zip(X, y):
		z = np.dot(xi, W)
		prediction = 1 if z > 0 else 0
		sprediction = "circle" if z > 0 else "square"
		correct = True if prediction == target else False
		print(f"prediction: [{sprediction}] activation: [{abs(z)}] correct: [{correct}]")

