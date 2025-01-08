Data Generation
[text](files_for_generating_data/Readme.md)

# Encoder for MNIST Digit Sum Model

## Overview

The encoder in this model is designed to process **grayscale images of size 40x168 pixels** containing 4 digits from the MNIST dataset. The encoder extracts feature maps through multiple **convolutional layers**, which are then passed through fully connected layers to output a latent representation of 40 features. This latent vector is reshaped into a probability distribution for each digit (0-9), which will later be processed by the decoder to compute the sum of the digits.

## Architecture Details

### 1. **Convolutional Blocks**

- **Number of Blocks**: 5 blocks
- **Each Block Contains**:
  - **Conv2D**: A 3x3 convolutional layer with **padding='same'** to preserve spatial dimensions.
  - **ReLU Activation**: To introduce non-linearity.
  - **Conv2D**: A second 3x3 convolutional layer with **padding='same'**.
  - **ReLU Activation**: Non-linearity again.
  - **MaxPool2D (2x2)**: Downsamples the feature maps by 2 in both spatial dimensions.
  - **Dropout (0.1)**: Randomly sets a fraction of input units to 0, reducing overfitting.

- **Channel Progression**: 
  - 1 → 64 → 128 → 256 → 512 → 1024

### 2. **Fully Connected Layers**

- **Flattened Output**: The output of the convolutional blocks is flattened into a 1D vector.
- **Dense Layer 1**: Fully connected layer with 512 units and **ReLU** activation.
- **Dropout (0.1)**: Dropout is applied after the fully connected layer to reduce overfitting.
- **Dense Layer 2**: Fully connected layer that outputs a 40-dimensional vector representing the latent features.

### 3. **Output Reshaping**

- The 40-dimensional output is reshaped into a **4x10 matrix**, where:
  - **4 digits** are predicted.
  - **10 probabilities per digit** (from 0 to 9) are calculated for each digit.

## Training Details

- **Loss Function**: **CrossEntropyLoss** is used for classification, where the output is treated as a probability distribution over the 10 possible classes for each digit.
- **Optimizer**: **Adam** optimizer with a learning rate of **0.00001**.
- **Checkpoint System**: The model saves checkpoints to ensure training continuity and resumption if needed.

## Data Pipeline

- Images are loaded in batches using **DataLoader** for efficient training.
- **Train/Test Split** is implemented for evaluating the model.
- **Shuffling** is enabled to ensure the model does not overfit to the order of the data.

## Full Data Flow

1. **Input**: The input image is of size **40x168 pixels**.
2. **Convolutional Blocks**: The image is passed through a series of convolutional layers, which extract high-level features.
3. **Feature Maps**: The resulting feature maps are processed by fully connected layers.
4. **Latent Features**: The fully connected layers output 40 features.
5. **Reshape**: The 40 features are reshaped into a **4x10 matrix**, where each of the 4 digits has 10 possible class probabilities.
6. **Decoder**: The final output is passed through a decoder that computes the sum of the digits.

### Encoder Summary

- **Purpose**: To convert the input image into a set of 4 digit probabilities, representing the sum of the digits in the image.
- **Architecture**: A stack of convolutional layers followed by fully connected layers.
- **Output**: 4 digits, each with a probability distribution over 10 possible values (0-9).



# MNIST Sum Decoder - Model Overview

## Overview

The **MNISTSumModel** decoder takes a 40-dimensional vector representing the probability distributions for four digits and computes their sum. The model is designed to predict the sum of digits with near-perfect accuracy. It achieves exceptionally low error on the test set and performs efficiently across a variety of sum values, demonstrating remarkable stability.

## Architecture Details

### Input

- **40-dimensional vector**: Representing the probability distributions for 4 digits (each digit having 10 possible classes).
  
### Network Structure

1. **Input Processing**:
   - Converts the input vector to float type.
   - Applies **softmax normalization** per digit to ensure the values sum to 1, representing probability distributions.
   - Reshapes the input to shape **(batch_size, 40)**.

2. **MLP Layers**:
   - **Linear + ReLU** (40 → 64): First fully connected layer followed by ReLU activation.
   - **Linear + ReLU** (64 → 128): Second fully connected layer with ReLU activation.
   - **Linear + ReLU** (128 → 64): Third fully connected layer with ReLU activation.
   - **Linear** (64 → 1): Final layer outputs a single scalar, representing the sum of the digits.

### Training Configuration

- **Loss Function**: **Mean Squared Error (MSE)** is used, as the task involves regression to predict the sum of digits.
- **Optimizer**: **Adam** optimizer with a learning rate of **0.001**.
- **Epochs**: The model is trained for **25 epochs**.

## Performance Results

### Test Set Performance

- **Test Loss**: **0.0002** (extremely low error, indicating high accuracy).

### Sample Predictions

- **Actual: 20** → **Predicted: 20.0027**
- **Actual: 23** → **Predicted: 23.0012**
- **Actual: 7** → **Predicted: 7.0282**
- **Actual: 19** → **Predicted: 19.0077**
- **Actual: 19** → **Predicted: 19.0017**
- **Actual: 27** → **Predicted: 27.0027**
- **Actual: 24** → **Predicted: 24.0052**
- **Actual: 5** → **Predicted: 5.0744**
- **Actual: 13** → **Predicted: 13.0038**
- **Actual: 14** → **Predicted: 14.0032**

### Key Observations

- **Average Prediction Error**: Less than 0.1, demonstrating precise sum prediction.
- **Consistent Performance**: The model performs consistently across a range of sum values.
- **Effectiveness for Small and Large Sums**: Accurate predictions are made even for small (5) and large (29) sums.
- **Stability**: The model shows remarkable stability in its predictions, with minimal variance from actual values.

## Model Strengths

- **High Accuracy**: Achieves minimal deviation from true sum values.
- **Efficient Architecture**: The model uses a compact architecture with only 4 layers.
- **Robust Performance**: Consistently performs well across different sum values, providing reliable predictions.


## Checkpoints

- **Checkpoint Storage**: Checkpoints are stored after running each **encoder** and **decoder** training file. These checkpoints capture the model state at various points during training, allowing for later recovery or fine-tuning.


- **Inference Notebook**: The **Inference Notebook** (`inference_notebook.ipynb`) is used for evaluation purposes. It automatically loads the latest checkpoint from encoder and decoder to make predictions and evaluate the model's performance on test data.[text](inference_notebook.ipynb)


- **Final Pipeline**: The training loop is implemented end-to-end and is intended for use with the assignment data. i.e. Image to sum [text](end_to_end.ipynb)

