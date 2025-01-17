import random
import os
from torchvision import datasets
from PIL import Image
import numpy as np
from tqdm import tqdm

# Parameters
mnist_data = datasets.MNIST(root="./mnist", train=True, download=True)
num_samples = 15000
image_width = 168
image_height = 40
horizontal_spacing = 1
output_dir = "dataset/train"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Normalization function for YOLO format
def normalize(value, max_value):
    return value / max_value

for idx in tqdm(range(num_samples), desc="Generating Images"):
    composite_image = Image.new("L", (image_width, image_height), color=0)
    annotation_lines = []

    # Number of digits fixed to 4 for each image
    num_digits = 4
    digits = random.choices(range(len(mnist_data)), k=num_digits)

    # Calculate the spacing for the digits (4 digits + 3 spaces)
    total_width = image_width - (num_digits - 1) * horizontal_spacing
    individual_width = total_width // num_digits

    current_x = 0  # Initial x-coordinate to place the first digit

    for digit_idx in digits:
        digit_image, label = mnist_data[digit_idx]

        # Random scaling of the digit
        scale = random.uniform(0.9, 1.2)
        new_width = int(digit_image.width * scale)
        new_height = int(digit_image.height * scale)
        digit_image = digit_image.resize((new_width, new_height))

        # Random rotation of the digit (within a range)
        rotation_angle = random.uniform(-15, 15)  # rotate between -15 to 15 degrees
        digit_image = digit_image.rotate(rotation_angle, expand=True)

        # Check if the rotated digit exceeds the image height
        if digit_image.height > image_height:
            # Skip this digit if it doesn't fit
            continue

        # Ensure the digit fits within the image height
        max_y = image_height - digit_image.height
        if max_y <= 0:
            # Skip if no valid space is available for the digit
            continue

        y_offset = random.randint(0, max_y)

        # Place digit at the calculated x-position
        if current_x + digit_image.width > image_width:
            break

        composite_image.paste(digit_image, (current_x, y_offset), digit_image)

        # YOLO format: class_id, x_center, y_center, width, height
        x_center = normalize(current_x + digit_image.width / 2, image_width)
        y_center = normalize(y_offset + digit_image.height / 2, image_height)
        width = normalize(digit_image.width, image_width)
        height = normalize(digit_image.height, image_height)
        annotation_lines.append(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Update x-coordinate for the next digit
        current_x += individual_width + horizontal_spacing

    # Save the image
    image_path = os.path.join(images_dir, f"{idx}.jpg")
    composite_image.save(image_path)

    # Save annotations
    if annotation_lines:
        label_path = os.path.join(labels_dir, f"{idx}.txt")
        with open(label_path, "w") as label_file:
            label_file.write("\n".join(annotation_lines))
