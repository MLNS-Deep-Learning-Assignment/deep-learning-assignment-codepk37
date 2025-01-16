
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image
# Custom Dataset class
class DigitSumDataset(Dataset):
    def __init__(self, data_files, label_files,transform ):
        # Load all data and labels
        self.data = []
        self.labels = []
        for data_file, label_file in zip(data_files, label_files):
            self.data.append(np.load(data_file))  # Load image data
            self.labels.append(np.load(label_file))  # Load labels

        # Combine all data and labels into a single array
        self.data = np.concatenate(self.data, axis=0)  # Shape: (N, H, W)
        self.labels = np.concatenate(self.labels, axis=0)  # Shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image.astype(np.uint8))  # Convert to 8-bit grayscale image

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, torch.zeros(40, dtype=torch.long), label,"--.png"


transform = transforms.Compose([
#    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Random rotation ±15° and shifts up to 10%
    transforms.Resize((40, 168)),  # Resize image to the correct size
    transforms.ToTensor(),         # Convert image to Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize (for grayscale images)
])

# File paths
data_files = ["data0.npy", "data1.npy", "data2.npy"]
label_files = ["lab0.npy", "lab1.npy", "lab2.npy"]

# Create the dataset
dataset = DigitSumDataset(data_files, label_files, transform=transform)
# Custom Dataset Class

print(len(dataset))
#from torch.utils.data import random_split
#train_size = int(0.1 * len(dataset)) #for now limited train images
#test_size = len(dataset) - train_size
# DataLoader for batching
#batch_size = 16
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
from torch.utils.data import DataLoader, Subset

# Assuming `dataset` is your dataset
dataset_size = len(dataset)
split = int(0.95 * dataset_size)  # 80% for training

# Define train and test indices based on order
train_indices = list(range(split))
test_indices = list(range(split, dataset_size))

# Use Subset to create train and test datasets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# DataLoader for batching
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# root_dirs = ["./exterim/images", "./exterim/images2"]
# mnist_dataset = MNISTDigitsDataset(root_dirs=root_dirs, transform=transform)

# # # Combine and Split
# combined_dataset = torch.utils.data.ConcatDataset([train_dataset, mnist_dataset])
# print(len(combined_dataset))

# train_dataset =combined_dataset
print(len(train_dataset))

# DataLoader for training and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Iterate over the DataLoader
for images,_ , labels,_ in train_loader:
    print("Images shape:", images.shape)  # (batch_size, C, H, W)
    print("Labels shape:", labels.shape)  # (batch_size,)
    print(np.unique(images[0]))
    break  # Process the first batch only


#class 1


import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTDigitModel(nn.Module):
    def __init__(self, num_blocks, kernel_size, activation, pool, dropout):
        super(MNISTDigitModel, self).__init__()
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool = pool
        self.dropout = dropout
        
        layers = []
        in_channels = 1  # Grayscale input images
        out_channels = 64  # Initial number of filters
        
        # Add convolutional blocks
        for _ in range(num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
                self._get_activation(activation),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
                self._get_activation(activation),
                self._get_pool(pool),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
            out_channels *= 2  # Double the filters after each block
            
        
        self.conv_blocks = nn.Sequential(*layers)
        
        # Dummy input to calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 40, 168)
            flattened_size = self.conv_blocks(dummy_input).numel()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(512, 40)  # 40 output classes (10 per digit for 4 digits)
        )
    
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Activation not supported")
    
    def _get_pool(self, pool):
        if pool == 'max':
            return nn.MaxPool2d(2)
        elif pool == 'avg':
            return nn.AvgPool2d(2)
        else:
            raise ValueError("Pooling method not supported")
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc(x)

        x = x.view(-1, 4, 10)
        return x





import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTSumModel(nn.Module):
    def __init__(self):
        super(MNISTSumModel, self).__init__()
        
        # MLP layers (flatten the 1x4x10 output)
        self.fc1 = nn.Linear(40, 64)  # 40 input features (4 digits * 10 classes)
        self.fc22 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single scalar (the sum of digits)

    def forward(self, x):
        # Apply softmax to each 10-length vector (per digit)
        x = x.float()
        x = F.softmax(x, dim=-1)
        
        # Flatten the input (4 digits * 10 classes)
        x = x.view(-1, 40)  # Shape becomes (batch_size, 40)

        # MLP layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc22(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Single scalar output
        return x
    



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Instantiate the models
new_dropout = 0.001
digit_model = MNISTDigitModel(num_blocks=5, kernel_size=3, activation='relu', pool='max', dropout=new_dropout)
sum_model = MNISTSumModel()


#for param in sum_model.parameters():
#    param.requires_grad = False

import os
# Define combined model
class CombinedModel(nn.Module):
    def __init__(self, digit_model, sum_model):
        super(CombinedModel, self).__init__()
        self.digit_model = digit_model
        self.sum_model = sum_model

    def forward(self, x):
        digit_output_encoded = self.digit_model(x)  # Shape: [batch_size, 4, 10]
        sum_output = self.sum_model(digit_output_encoded)  # Shape: [batch_size, 1]
        return digit_output_encoded, sum_output

combined_model = CombinedModel(digit_model, sum_model)



# Optimizer and criterion
new_learning_rate = 0.00001 ##0.0001 for CNN
optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=new_learning_rate)
# optimizer = optim.Adam([
#     {'params': filter(lambda p: p.requires_grad, digit_model.parameters()), 'lr': 0.00001},
# ])

# optimizer = optim.Adam([
#     {'params': digit_model.parameters(), 'lr': 0.00001},  # Learning rate for digit_model
#     {'params': sum_model.parameters(), 'lr': 0.0},  # Frozen sum_model (effectively ignored)
# ])

# Load the combined model checkpoint
checkpoint_path = './checkpoints_comb/checkpoint_epoch_700.pth'
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"Loading combined model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    # Load the combined model weights
    combined_model.load_state_dict(checkpoint['model_state_dict'])
    # Set the starting epoch for resuming training
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}...")

# criterion_digit = nn.CrossEntropyLoss()  # For digit classification
criterion_sum = nn.MSELoss()  # For sum regression (used only for evaluation)


# Training loop for combined model
num_epochs = 10000
checkpoint_dir = './checkpoints_comb'  # Directory for saving checkpoints
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the checkpoint directory exists

for epoch in range(start_epoch, start_epoch + num_epochs):
    combined_model.train()  # Set model to training mode
    # running_loss_digit = 0.0
    # running_loss_sum = 0.0
    running_loss = 0.0
    for images, _, labels_sum, _ in train_loader:  # Adjust based on your dataset
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        
        _, sum_output = combined_model(images)
        # sum_output = sum_output.view(-1) 
        # Compute loss (compare output to label_num)
        loss = criterion_sum(sum_output.view(-1), labels_sum.float())  # Squeeze to remove extra dimensions
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': combined_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    if True:
        combined_model.eval()
        # Evaluation on test_loader
        with torch.no_grad():  # Disable gradient computation for evaluation
            for images, _, labels_sum, nam in test_loader:  # Adjust based on your dataset
                # Forward pass
                _, sum_output = combined_model(images)


                # Optionally, compare predicted and actual values
                for i in range(3):
                    print(f"Sample :",nam[i])
                    print(f"Original Sum: {labels_sum[i]}")
                    print(f"Predicted Sum: {sum_output[i].item()}")  # Use .item() to get the scalar value

                break  # Just evaluate the first batch and exit the loop
