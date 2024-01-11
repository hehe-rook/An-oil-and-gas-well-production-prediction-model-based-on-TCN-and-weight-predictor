import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read CSV file
data = pd.read_csv('1.csv')

# Select the target column for prediction
target_col = 'Daily gas capacity(m3)'
target_data = data[target_col].values.reshape(-1, 1)

# Perform feature scaling
scaler = MinMaxScaler()
target_data = scaler.fit_transform(target_data)

# Save the scaled data to a new CSV file
scaled_data = pd.DataFrame(target_data, columns=[target_col])
scaled_data.to_csv('scaled_data.csv', index=False)
