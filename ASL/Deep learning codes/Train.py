import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchvision import transforms

sys.setrecursionlimit(1500)

class TimeHistory:
    def __init__(self):
        self.times = []
    
    def start_epoch(self):
        self.epoch_time_start = time.time()
    
    def end_epoch(self):
        self.times.append(time.time() - self.epoch_time_start)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNNLSTMModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.4)  # Dropout after LSTM layer
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)  # Added one more layer before output
        self.fc_out = nn.Linear(1024, output_dim)  # Updated final output layer
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for CNN
        
        # Convolutional layers with BatchNorm
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = x.permute(0, 2, 1)  # Reshape for LSTM
        
        # LSTM layers with Dropout
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        
        # Fully connected layers with Dropout
        x = self.relu(self.fc1(x))  
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))  # Apply the new layer
        x = self.dropout(x)
        x = torch.sigmoid(self.fc_out(x))
        
        return x

def data_generator():
    """Generate training and validation datasets along with input dimensions."""
    with open('train_data.pkl', 'rb') as data:
        dataset = pickle.load(data)
    with open('valid_data.pkl', 'rb') as data_valid:
        dataset_valid = pickle.load(data_valid)
    
    enc = LabelEncoder()
    dataset['Y'] = enc.fit_transform(dataset['Y'])
    ohe = OneHotEncoder().fit(dataset['Y'].reshape(-1, 1))
    dataset['Y'] = ohe.transform(dataset['Y'].reshape(-1, 1)).toarray()
    dataset_valid['Y'] = enc.transform(dataset_valid['Y'])
    dataset_valid['Y'] = ohe.transform(dataset_valid['Y'].reshape(-1, 1)).toarray()
    
    return (
        CustomDataset(dataset['X'], dataset['Y']),
        CustomDataset(dataset_valid['X'], dataset_valid['Y']),
        dataset['X'].shape[1],
        len(ohe.categories_[0])
    )

def apply_transforms(X):
    """ Apply transformations to input data before passing to the model """
    # For now, we simply normalize the data (you can add augmentations here)
    return X / 255.0  # Assuming the input is pixel values ranging from 0 to 255

def train_model(epochs=60):
    """Train the CNN-LSTM model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset, valid_dataset, input_dim, num_classes = data_generator()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    
    model = CNNLSTMModel(input_dim=input_dim, hidden_dim=64, output_dim=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)  # Updated optimizer to AdamW
    
    # Learning rate scheduler (StepLR with gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "time": []}
    time_callback = TimeHistory()
    
    for epoch in range(epochs):
        model.train()
        time_callback.start_epoch()
        train_loss, correct, total = 0, 0, 0
        
        for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            # Apply transforms before feeding data to the model
            X_batch = apply_transforms(X_batch)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(Y_batch, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        history["loss"].append(train_loss / len(train_loader))
        history["accuracy"].append(correct / total)
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, Y_batch in valid_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
                # Apply transforms before feeding data to the model
                X_batch = apply_transforms(X_batch)
                
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(Y_batch, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        history["val_loss"].append(val_loss / len(valid_loader))
        history["val_accuracy"].append(val_correct / val_total)
        time_callback.end_epoch()
        history["time"].append(time_callback.times[-1])
        
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {correct / total:.4f}")
        print(f"  Val Loss: {val_loss / len(valid_loader):.4f}, Val Accuracy: {val_correct / val_total:.4f}")
        
        scheduler.step()
    
    torch.save(model.state_dict(), "sequential_model.pth")
    with open("sequential_model_score.pkl", "wb") as f:
        pickle.dump(history, f)
    print("Training completed!")

if __name__ == "__main__":
    train_model()
