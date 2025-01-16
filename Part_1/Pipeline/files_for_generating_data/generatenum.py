import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import cv2
import random
import os
from concurrent.futures import ThreadPoolExecutor

# Set up directories
output_img_dir = "exterim/images"
output_txt_dir = "exterim/label"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_txt_dir, exist_ok=True)

# Load the MNIST dataset
transform = transforms.ToTensor()
mnist = MNIST(root="./data", train=True, download=True, transform=transform)

# Function to randomly pick a 4-digit number and select corresponding digits from the dataset
def get_digits_for_random_number():
    random_number = random.randint(0, 9999)
    digits = [int(digit) for digit in str(random_number).zfill(4)]  # Ensure it has 4 digits (e.g., 0042)
    print(digits)
    # Select corresponding digits from MNIST dataset
    selected_digits = []
    for digit in digits:
        digit_indices = [i for i, (_, label) in enumerate(mnist) if label == digit]
        if digit_indices:
            selected_idx = random.choice(digit_indices)
            selected_digits.append(mnist[selected_idx][0].numpy().squeeze())  # Get the image of the selected digit
        else:
            selected_digits.append(np.zeros((28, 28)))  # Default to an empty digit if not found

    return random_number, selected_digits

# Function to apply translation and rotation to the digit pixels
def transform_digit_pixels(digit, canvas_size, x_offset):
    digit_height, digit_width = digit.shape
    canvas_height, canvas_width = canvas_size
    
    # Get coordinates of non-zero pixels (digit pixels only)
    y_coords, x_coords = np.where(digit > 0)
    pixel_values = digit[y_coords, x_coords]  # Extract pixel intensities
    
    # Center the digit for transformation
    x_center, y_center = digit_width / 2, digit_height / 2
    x_coords_centered = x_coords - x_center
    y_coords_centered = y_coords - y_center
    
    # Random rotation
    angle = random.uniform(-15, 15)  # Rotate within [-15, 15] degrees
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    coords_rotated = np.dot(
        rotation_matrix[:, :2],
        np.vstack((x_coords_centered, y_coords_centered))
    )
    x_coords_rotated, y_coords_rotated = coords_rotated[0], coords_rotated[1]
    
    # Random translation (shift along x and y)
    tx = random.randint(-10, 10)
    ty = random.randint(0, canvas_height - digit_height - 1)  # Ensure fit within canvas
    
    x_coords_translated = x_coords_rotated + x_center + x_offset + tx
    y_coords_translated = y_coords_rotated + y_center + ty
    
    # Clip coordinates to stay within canvas
    x_coords_final = np.clip(x_coords_translated, 0, canvas_width - 1).astype(int)
    y_coords_final = np.clip(y_coords_translated, 0, canvas_height - 1).astype(int)
    
    return x_coords_final, y_coords_final, pixel_values

# Function to process a single random number and its digits
def process_random_number():
    random_number, digits = get_digits_for_random_number()
    
    # Create a blank canvas
    canvas_height, canvas_width = 40, 168
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    current_x = 0  # Starting x-offset
    for digit in digits:
        digit_resized = cv2.resize(digit, (24, 28), interpolation=cv2.INTER_AREA)
        
        # Transform the digit pixels and inject into canvas
        x_coords, y_coords, pixel_values = transform_digit_pixels(digit_resized, (canvas_height, canvas_width), current_x)
        
        # Ensure that we only update empty (zero) pixels on the canvas
        mask = canvas[y_coords, x_coords] == 0
        canvas[y_coords[mask], x_coords[mask]] = pixel_values[mask]
        
        # Update current_x for next digit placement
        current_x += digit_resized.shape[1] + random.randint(5, 15)
    
    random_number_str = str(random_number).zfill(4)  # Zero pad to ensure it's 4 digits
    img_filename = os.path.join(output_img_dir, f"{random_number_str}.png")
    txt_filename = os.path.join(output_txt_dir, f"{random_number_str}.txt")
    
    # Save the image
    cv2.imwrite(img_filename, (canvas * 255).astype(np.uint8))
    
    # Save the random number to a text file
    with open(txt_filename, 'w') as f:
        f.write(f"{random_number_str}\n")
    
    print(f"Image and label saved for {random_number}")

# Parallelize the process using a ThreadPoolExecutor
def main():
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit tasks to the executor for parallel processing
        futures = [executor.submit(process_random_number) for _ in range(3000)]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

# Run the script
if __name__ == "__main__":
    main()
