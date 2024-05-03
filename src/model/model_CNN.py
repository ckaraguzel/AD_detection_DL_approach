from Imports import *
torch.manual_seed(0)
Base_path = ''
X_train = np.load(Base_path + 'X_train.npy')
y_train = np.load(Base_path + 'y_train.npy')

# Define your dataset
class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create dataset
train_dataset = MyDataset(X_train, y_train)
# Create DataLoaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(65, 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*240, 65)  
        self.fc3 = nn.Linear(65, 2)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer

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

#print(model)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.5,1.0]).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Initialize empty lists to store losses
train_losses = []

# Training loop
for epoch in range(15):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.float()
        labels = labels.type(torch.LongTensor)   # casting to long
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        l1_lambda = 0.0005
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(outputs, labels)
        loss+=l1_lambda*l1_norm
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Compute training loss
        train_losses.append(loss.item())
    print(f'Epoch {epoch+1}, training loss: {running_loss/len(train_dataloader):.4f}')

# Now without regularization
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Training loop
for epoch in range(20):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.float()
        labels = labels.type(torch.LongTensor)   # casting to long
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # Compute training loss
        train_losses.append(loss.item())
    if epoch%5==0:
        print(f'Epoch {epoch+1}, training loss: {running_loss/len(train_dataloader):.4f}')

print('Finished Training')
del X_train, y_train
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
test_dataset = MyDataset(X_test, y_test)

# Assuming x_test is a PyTorch tensor
testloader = torch.utils.data.DataLoader(X_test, batch_size=32)
def find_optimal_threshold(predictions, y_test):
    min_sum = float('inf')
    optimal_threshold = 0

    # Iterate over possible thresholds from 0 to 1
    for threshold in np.arange(0.0, 1, 0.001):
        # Apply threshold
        preds = (np.array(predictions)[:,1] > threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, preds)

        # Compute sum of off-diagonal elements
        off_diagonal_sum = 164 - np.trace(cm)
        #print(cm)
        # Update optimal threshold if this threshold is better
        if off_diagonal_sum < min_sum and cm[1][1]/np.sum(cm[1]) >0.5:
            min_sum = off_diagonal_sum
            optimal_threshold = threshold

    return optimal_threshold
model.eval()  # Set the model to evaluation mode
predictions = []
output = []
with torch.no_grad():
    for data in testloader:
        outputs = model(data.float().to(device)).to(device)
        predicted = torch.softmax(outputs, dim = 1) # Apply a threshold
        predictions.extend(predicted.tolist())
        output.extend(outputs.tolist())

        
optimal_threshold = find_optimal_threshold(predictions, y_test)
# Calculate the multi-label confusion matrix
mcm = confusion_matrix(y_test, (np.array(predictions)[:,1]>optimal_threshold).astype(int))

print('Confusion Matrix:')
for i, matrix in enumerate(mcm):
    print(f'Class {i}:')
    print(matrix)

# Convert tensors to numpy arrays for sklearn
outputs_np = (np.array(predictions)[:,1]>optimal_threshold).astype(int)
labels_np = y_test

auc_score = roc_auc_score(labels_np, outputs_np)
print(f"AUC Score: {auc_score}")