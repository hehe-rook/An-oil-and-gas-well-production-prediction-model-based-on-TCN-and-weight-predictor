import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from TCN import TCN
from WeightPredictor import WeightPredictor

# Read CSV file
data = pd.read_csv('1.csv')

# Select the target column for prediction
target_col = 'Daily gas capacity(m3)'
target_data = data[target_col].values.reshape(-1, 1)

# Perform feature scaling
scaler = MinMaxScaler()
target_data = scaler.fit_transform(target_data)

# Define hyperparameters
input_size = 1  # Number of input features
output_size = n  # Number of output features
num_channels = [64, 128, 256]  # Number of channels for each level in the TCN model
kernel_sizes = [3, 5, 7]  # Convolutional kernel sizes in the TCN model
dropout = 0.2  # Dropout rate in the TCN model
hidden_size = 128  # Hidden size in the weight predictor model
lr = 0.001  # Learning rate
epochs = 100  # Number of training epochs

# Convert data to PyTorch tensor
target_tensor = torch.Tensor(target_data).unsqueeze(1)

# Split the data into training and test sets
train_size = int(len(target_tensor) * 0.8)
train_data = target_tensor[:train_size]
test_data = target_tensor[train_size:]

# Initialize TCN and weight predictor models
tcn_model = TCN(input_size, output_size, num_channels, kernel_sizes, dropout)
weight_predictor = WeightPredictor(input_size, hidden_size)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(list(tcn_model.parameters()) + list(weight_predictor.parameters()), lr=lr)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Generate predictions using the TCN model
    tcn_output = tcn_model(train_data)

    # Generate weights using the weight predictor model
    weights = weight_predictor(train_data)

    # Weighted sum of the predictions
    weighted_output = torch.sum(tcn_output * weights, dim=1)

    # Compute the loss
    loss = criterion(weighted_output, train_data[:, 0])

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

    # Make predictions on the test set
    with torch.no_grad():
        tcn_model.eval()
        weight_predictor.eval()

        tcn_output = tcn_model(test_data)
        weights = weight_predictor(test_data)
        weighted_output = torch.sum(tcn_output * weights, dim=1)

        # Convert the data back to the original scale
        predicted_data = scaler.inverse_transform(weighted_output.squeeze(1).numpy().reshape(-1, 1))
        print("Predicted data:")
        print(predicted_data)
