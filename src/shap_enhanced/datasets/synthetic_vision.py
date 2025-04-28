# TODO: Move to CPP

import torch
import numpy as np
import cv2

def draw_circle(image, center, radius, color):
    """Draw a circle on the image."""
    cv2.circle(image, center, radius, color, -1)

def draw_square(image, top_left, size, color):
    """Draw a square on the image."""
    bottom_right = (top_left[0] + size, top_left[1] + size)
    cv2.rectangle(image, top_left, bottom_right, color, -1)

def draw_triangle(image, vertices, color):
    """Draw a triangle on the image."""
    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], color)

def generate_vision_data(n_samples: int, image_size: int = 32, n_classes: int = 2, shapes=None):
    """Generate synthetic images with basic shapes.

    :param int n_samples: Number of samples to generate.
    :param int image_size: Size of the square image (image_size x image_size), defaults to 32
    :param int n_classes: Number of classes, defaults to 2
    :param shapes: List of shapes to include (e.g., ['circle', 'square', 'triangle']), defaults to None
    :return torch.Tensor: Tensor of images and labels.
    """
    if shapes is None:
        shapes = ['circle', 'square']  # Default shapes

    X = np.zeros((n_samples, 3, image_size, image_size), dtype=np.float32)  # RGB images
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        image = np.zeros((image_size, image_size, 3), dtype=np.float32)  # Single RGB image
        shape_type = np.random.choice(shapes)
        label = shapes.index(shape_type) % n_classes  # Map shape to class label
        y[i] = label

        color = tuple(np.random.rand(3))  # Random RGB color
        if shape_type == 'circle':
            center = (np.random.randint(10, image_size - 10), np.random.randint(10, image_size - 10))
            radius = np.random.randint(5, min(image_size // 4, 15))
            draw_circle(image, center, radius, color)
        elif shape_type == 'square':
            top_left = (np.random.randint(5, image_size - 15), np.random.randint(5, image_size - 15))
            size = np.random.randint(5, min(image_size // 4, 15))
            draw_square(image, top_left, size, color)
        elif shape_type == 'triangle':
            vertices = [
                (np.random.randint(5, image_size - 5), np.random.randint(5, image_size - 5)),
                (np.random.randint(5, image_size - 5), np.random.randint(5, image_size - 5)),
                (np.random.randint(5, image_size - 5), np.random.randint(5, image_size - 5))
            ]
            draw_triangle(image, vertices, color)

        X[i] = image.transpose(2, 0, 1)  # Convert to channel-first format

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
