
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
    
class CombinedModel(nn.Module):
    def __init__(self, digit_model, sum_model):
        super(CombinedModel, self).__init__()
        self.digit_model = digit_model
        self.sum_model = sum_model

    def forward(self, x):
        digit_output_encoded = self.digit_model(x)  # Shape: [batch_size, 4, 10]
        sum_output = self.sum_model(digit_output_encoded)  # Shape: [batch_size, 1]
        return digit_output_encoded, sum_output
