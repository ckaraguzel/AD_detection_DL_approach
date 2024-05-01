# Import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(42)
np.random.seed(42)

# Define your custom dataset
class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# Simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(65, 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*240, 65) 
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer
        self.fc3 = nn.Linear(65, 2)

    def forward(self, x):
        x = x.view(-1, 65, 256, 240)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
# Initialize the network and print its architecture
model = Net()
print(model)

# Extract the features and labels from the training data
X_train = df_train[:,3]
y_train = df_train[:,2]

# Create a dataset from the training data
train_dataset = MyDataset(X_train, y_train)

# Create a DataLoader for the training data
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Delete the original training features to save memory
del X_train

# Extract the features and labels from the test and validation data
X_test = df_test[:,3]
y_test = df_test[:,2]
X_val = df_validation[:,3]
y_val = df_validation[:,2]

# Create datasets from the validation and test data
val_dataset = MyDataset(X_val, y_val)
testloader = torch.utils.data.DataLoader(X_test, batch_size=32)

# Initialize the network
model = Net()
torch.manual_seed(42)
np.random.seed(42)
# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.5,1.0]).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Initialize a list to store the training losses
train_losses = []

# Training loop
for epoch in range(15):
    running_loss = 0.0
    model.train()  # Set the model to training mode
    for i, data in enumerate(train_dataloader, 0):
        # Get the inputs and labels from the current batch of data
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.float()  # Convert the inputs to float
        labels = labels.type(torch.LongTensor)   # Cast the labels to long
        inputs, labels = inputs.to(device), labels.to(device)  # Move the inputs and labels to the device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the L1 norm of the model parameters
        l1_lambda = 0.0005
        l1_norm = sum(p.abs().sum() for p in model.parameters())

        # Compute the loss
        loss = criterion(outputs, labels)
        loss += l1_lambda * l1_norm  # Add the L1 regularization term to the loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

        # Store the loss for this batch
        train_losses.append(loss.item())

    scheduler.step()
    # Print the average training loss for this epoch
    print(f'Epoch {epoch+1}, training loss: {running_loss/len(train_dataloader):.4f}')

# Now train the model without L1 regularization
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Training loop
for epoch in range(15):
    running_loss = 0.0
    model.train()  # Set the model to training mode
    for i, data in enumerate(train_dataloader, 0):
        # Get the inputs and labels from the current batch of data
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.float()  # Convert the inputs to float
        labels = labels.type(torch.LongTensor)   # Cast the labels to long
        inputs, labels = inputs.to(device), labels.to(device)  # Move the inputs and labels to the device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

        # Store the loss for this batch
        train_losses.append(loss.item())

    # Print the average training loss for this epoch
    print(f'Epoch {epoch+1}, training loss: {running_loss/len(train_dataloader):.4f}')

# Print a message to indicate that training is finished
print('Finished Training')



# Define a function to find the optimal threshold for classification
def find_optimal_threshold(predictions, y_test):
    min_sum = float('inf')  # Initialize the minimum sum of off-diagonal elements
    optimal_threshold = 0.5  # Initialize the optimal threshold
    # Iterate over possible thresholds from 0.3 to 0.8
    for threshold in np.arange(0.3, 0.8, 0.001):
        # Apply threshold to the predictions
        preds = (np.array(predictions)[:,1] > threshold).astype(int)
        # Compute confusion matrix for the current threshold
        cm = confusion_matrix(y_test, preds)
        # Compute sum of off-diagonal elements of the confusion matrix
        off_diagonal_sum = 164 - np.trace(cm)
        # Update optimal threshold if this threshold gives a lower sum of off-diagonal elements
        if off_diagonal_sum < min_sum and cm[1][1]/np.sum(cm[1]) >=0.5:
            min_sum = off_diagonal_sum
            optimal_threshold = threshold
    # Return the optimal threshold
    return optimal_threshold

# Set the model to evaluation mode
model.eval()

# Initialize lists to store predictions and outputs
predictions = []
output = []

# Make predictions on the validation data
with torch.no_grad():
    for data in valloader:
        outputs = model(data[0].float().to(device)).to(device)
        predicted = torch.softmax(outputs, dim = 1)  # Apply softmax to the outputs
        predictions.extend(predicted.tolist())  # Store the predictions
        output.extend(outputs.tolist())  # Store the outputs

# Find the optimal threshold on the validation data
optimal_threshold = find_optimal_threshold(predictions, y_val.astype(int))

# Compute and print the confusion matrix for the validation data
mcm = confusion_matrix(y_val.astype(int), (np.array(predictions)[:,1]>optimal_threshold).astype(int))
print('Confusion Matrix:')
for i, matrix in enumerate(mcm):
    print(f'Class {i}:')
    print(matrix)

# Reset the predictions and outputs lists
predictions = []
output = []

# Make predictions on the test data
with torch.no_grad():
    for data in testloader:
        outputs = model(data.float().to(device)).to(device)
        predicted = torch.softmax(outputs, dim = 1)  # Apply softmax to the outputs
        predictions.extend(predicted.tolist())  # Store the predictions
        output.extend(outputs.tolist())  # Store the outputs

# Compute and print the confusion matrix for the test data
mcm = confusion_matrix(y_test.astype(int), (np.array(predictions)[:,1]>optimal_threshold).astype(int))
print('Confusion Matrix:')
for i, matrix in enumerate(mcm):
    print(f'Class {i}:')
    print(matrix)

# Import the roc_auc_score function from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Convert the predictions and labels to numpy arrays for the roc_auc_score function
outputs_np = (np.array(predictions)[:,1]>optimal_threshold).astype(int)
labels_np = y_test.astype(int)

# Compute and print the AUC score
auc_score = roc_auc_score(labels_np, outputs_np)
print(f"AUC Score: {auc_score}")
