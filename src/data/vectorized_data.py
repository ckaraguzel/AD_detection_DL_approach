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
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
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

# Using the previously trained model to get vectorized data
class MyModel(nn.Module):
    def __init__(self, original_model):
        super(MyModel, self).__init__()
        self.conv1 = original_model.conv1
        self.fc1 = original_model.fc1

    def forward(self, x):
        x = x.view(-1, 65, 256, 240)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        return x

# Create the new model
new_model = MyModel(model)

outputs = [] # Initialize an empty list to store the output features from the model
labels_train = [] # Initialize an empty list to store the training labels
for data in train_dataloader: # Loop over the training data
    input_data = data[0].float() # Convert the input data to float
    labels_train.append(data[1]) # Append the labels to the labels_train list
    output = new_model(input_data.to(device)) # Pass the input data through the model
    outputs.append(output.detach().cpu().numpy()) # Detach the output from the computation graph, move it to cpu and convert it to numpy array, then append it to the outputs list
flattened_outputs = np.concatenate(outputs, axis=0) # Concatenate the outputs along axis 0 to flatten them
flattened_labels = np.concatenate(labels_train, axis = 0) # Concatenate the labels along axis 0 to flatten them
del outputs, output # Delete the outputs and output variables to free up memory
np.save('vectorized_train.npy', flattened_outputs) # Save the flattened outputs to a numpy file
np.save('vectorized_tr_labels.npy', flattened_labels) # Save the flattened labels to a numpy file

# Repeat the same process for the validation data
outputs = [] 
labels_val = []
for data in valloader:
    input_data = data[0].float()
    labels_val.append(data[1])
    output = new_model(input_data.to(device))
    outputs.append(output.detach().cpu().numpy()) 
flattened_outputs = np.concatenate(outputs, axis=0)
flattened_labels = np.concatenate(labels_val, axis = 0)
del outputs, output
np.save('vectorized_val.npy', flattened_outputs)
np.save('vectorized_val_labels.npy', flattened_labels)

# Repeat the same process for the test data, but without labels
outputs = [] 
for data in testloader:
    input_data = data.float()
    output = new_model(input_data.to(device))
    outputs.append(output.detach().cpu().numpy()) 
flattened_outputs = np.concatenate(outputs, axis=0)
del outputs, output
np.save('vectorized_test.npy', flattened_outputs)
np.save('y_test.npy', y_test.astype(int)) # Save the test labels to a numpy file
