#!/usr/bin/env python3

# image generator for simple perceptron

from PIL import Image, ImageDraw
import random
import os

IMG_SIZE = 50

def generate_images(count, shape_type, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(count):
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=0)
        draw = ImageDraw.Draw(img)

        size = random.randint(10, 25) 

        pos_x = random.randint(0, IMG_SIZE - size)
        pos_y = random.randint(0, IMG_SIZE - size)
        
        if shape_type == 'circle':
            draw.ellipse([(pos_x, pos_y), (pos_x + size, pos_y + size)], fill=255)
        elif shape_type == 'square':
            draw.rectangle([(pos_x, pos_y), (pos_x + size, pos_y + size)], fill=255)
            
        img.save(f"{path}/{shape_type[0]}{i}.jpg")

generate_images(200, 'circle', 'simple/circles')
generate_images(200, 'square', 'simple/squares')
generate_images(10, 'circle', 'simple/test')
generate_images(10, 'square', 'simple/test')
