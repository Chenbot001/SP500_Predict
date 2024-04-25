#%% 
# Setup
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from lstmDataset import lstmDataset
from lstmModel import LSTM
from sklearn.preprocessing import MinMaxScaler

# Functions
def n_days_history(data_series, lookback) -> pd.DataFrame:

    df = pd.DataFrame(data_series, columns=['Close'])
    for day in range(lookback):
        df[f'Close(t-{day+1})'] = df['Close'].shift(day+1)

    last_n_days = df[lookback:]

    return last_n_days

def train_one_epoch() -> None:
    model.train(True)
    print(f'Epoch: {epoch + 1}')

    
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        output = model(x_batch)
        loss = loss_fn(output, y_batch)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss.append(loss.item())

def validate_one_epoch() -> None:
    model.train(False)
    running_loss = 0.0
    
    for _, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')

def test_model(test_data,ground_truth):
    with torch.no_grad():
        test_predictions = model(test_data.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((test_data.shape[0], lookback+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dummies[:, 0]

    dummies = np.zeros((test_data.shape[0], lookback+1))
    dummies[:, 0] = ground_truth.cpu().flatten()
    dummies = scaler.inverse_transform(dummies)
    new_truth = dummies[:, 0]

    plt.plot(new_truth, label='Actual Close')
    plt.plot(test_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.title('Pred vs Actual')
    plt.legend()
    plt.show()
    print("Error: ",np.sqrt(np.mean((test_predictions - new_truth) ** 2)).item())

#%% 
# Data preprocessing
SPYdata = yf.download('SPY', start='1999-12-31', end=date.today()+timedelta(days=1))

# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running on',device)

# format data for model use
close_data = SPYdata[['Close']]
lookback = 7
df_data = n_days_history(close_data,lookback)
np_data = df_data.to_numpy()
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_data = scaler.fit_transform(np_data)

X = normalized_data[:, 1:]
y = normalized_data[:, 0]

# train-test split for time series
train_size = int(len(normalized_data) * 0.8)


X_train = torch.tensor(X[:train_size],dtype=torch.float32).unsqueeze(-1).to(device)
y_train = torch.tensor(y[:train_size],dtype=torch.float32).unsqueeze(-1).to(device)

X_test = torch.tensor(X[train_size:],dtype=torch.float32).unsqueeze(-1).to(device)
y_test = torch.tensor(y[train_size:],dtype=torch.float32).unsqueeze(-1).to(device)

# create datasets
train_data = lstmDataset(X_train,y_train)
test_data = lstmDataset(X_test,y_test)

#%% 
# Model
input_size = 1
hidden_size = 100
num_layers = 2
train_loader = data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=16, shuffle=False)

# Create model instance
model = LSTM(input_size, hidden_size, num_layers).to(device)

#%%
# Training

loss_fn = nn.MSELoss()
n_epochs = 10
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loss = []

for epoch in range(n_epochs):
    train_one_epoch()
    validate_one_epoch()
    
plt.plot(range(1, n_epochs+1), train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()


# Test and Plot
test_model(X_train,y_train)
test_model(X_test,y_test)
#%% Test other stocks

# %%
