import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import cv2

# Load the MNIST dataset
transform = transforms.ToTensor()
mnist = MNIST(root="./data", train=True, download=True, transform=transform)

# Select 4 random digits
indices = torch.randint(0, len(mnist), (4,))
digits = [mnist[i][0].numpy().squeeze() for i in indices]

# Resize each digit to a smaller size (35x38)
smaller_digits = [cv2.resize(d, (38, 35), interpolation=cv2.INTER_AREA) for d in digits]

# Create blank images of size (40x42) for each digit
centered_digits = []
for d in smaller_digits:
    blank = np.zeros((40, 42), dtype=np.float32)  # Create a blank canvas
    # Calculate the offsets to center the smaller digit
    x_offset = (42 - 38) // 2
    y_offset = (40 - 35) // 2
    blank[y_offset:y_offset + 35, x_offset:x_offset + 38] = d
    centered_digits.append(blank)

# Concatenate digits horizontally
combined_image = np.hstack(centered_digits)

# Save or display the image
output_path = "mnist_4_digits_smaller.png"
cv2.imwrite(output_path, (combined_image * 255).astype(np.uint8))
print(f"Image saved to {output_path}")





import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
import cv2
import random
import os
from concurrent.futures import ThreadPoolExecutor

# Set up directories
output_img_dir = "exterim/images2"
output_txt_dir = "exterim/label2"
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
    
    # Resize each digit to a smaller size (35x38)
    smaller_digits = [cv2.resize(d, (38, 35), interpolation=cv2.INTER_AREA) for d in digits]

    # Create blank images of size (40x42) for each digit
    centered_digits = []
    for d in smaller_digits:
        blank = np.zeros((40, 42), dtype=np.float32)  # Create a blank canvas
        # Calculate the offsets to center the smaller digit
        x_offset = (42 - 38) // 2
        y_offset = (40 - 35) // 2
        blank[y_offset:y_offset + 35, x_offset:x_offset + 38] = d
        centered_digits.append(blank)

    # Concatenate digits horizontally
    combined_image = np.hstack(centered_digits)

    random_number_str = str(random_number).zfill(4)  # Zero pad to ensure it's 4 digits
    img_filename = os.path.join(output_img_dir, f"{random_number_str}.png")
    txt_filename = os.path.join(output_txt_dir, f"{random_number_str}.txt")
    

    # Save the image
    cv2.imwrite(img_filename, (combined_image * 255).astype(np.uint8))
    
    # Save the random number to a text file
    with open(txt_filename, 'w') as f:
        f.write(f"{random_number_str}\n")
    
    print(f"Image and label saved for {random_number}")

# Parallelize the process using a ThreadPoolExecutor
def main():
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit tasks to the executor for parallel processing
        futures = [executor.submit(process_random_number) for _ in range(30)]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()

# Run the script
if __name__ == "__main__":
    main()
