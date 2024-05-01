import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

# Load the training data
X_train = np.load('probs_train.npy')
y_train = np.load('tr_labels.npy')

# Load the validation data
X_val = np.load('probs_val.npy')
y_val = np.load('val_labels.npy')

# Load the test data
X_test = np.load('probs_test.npy')
y_test = np.load('y_test.npy')

# Save the original np.load function
np_load_old = np.load

# Modify the default parameters of np.load to allow loading pickled data
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Load the additional dataframes
df_train = np.load('df_train.npy')
df_validation = np.load('df_validation.npy')
df_test = np.load('df_test.npy')

# Restore np.load for future normal usage
np.load = np_load_old

# Convert the 'sex' column from string to int
# 'M' is converted to 1 and 'F' is converted to 0
df_train[:,0] = (df_train[:,0]=='M').astype(int)
df_validation[:,0] = (df_validation[:,0]=='M').astype(int)
df_test[:,0] = (df_test[:,0]=='M').astype(int)

# Add the 'sex' and 'age' columns to your X_train, X_val, and X_test
# This enriches the feature set with demographic information
X_train = np.hstack((X_train, df_train[:,0:2]))
X_val = np.hstack((X_val, df_validation[:,0:2]))
X_test = np.hstack((X_test, df_test[:,0:2]))

# Import the necessary library
from sklearn.model_selection import RandomizedSearchCV

# Set a seed for reproducibility
np.random.seed(42)

# Stack the training and validation data and labels
data = np.vstack((X_train,X_val))
labels = np.hstack((y_train,y_val))

# Split the data into training and testing sets
# The test size is 30% of the total data
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Calculate the ratio of negative to positive instances in the training data
scale_pos_weight = sum(labels_train == 0) / sum(labels_train == 1)

# Define the XGBoost model with the objective function for binary classification
# The scale_pos_weight parameter helps handle class imbalance
# Regularization is added through the 'reg_alpha' and 'reg_lambda' parameters
model = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight = scale_pos_weight,
    random_state=42,
    reg_alpha = 0.1,  # L1 regularization term on weight (analogous to Lasso regression)
    reg_lambda = 1  # L2 regularization term on weight (analogous to Ridge regression)
)

# Define the parameter grid for RandomSearch
param_grid = {
    'max_depth': [2, 3, 5, 7],  # Maximum depth of the trees
    'learning_rate': [0.001, 0.002, 0.005, 0.01],  # Learning rate (eta)
    'n_estimators': [50, 100, 200],  # Number of training rounds
    'reg_alpha': [0.1, 0.5, 1],  # L1 regularization term on weight
    'reg_lambda': [0.1, 0.5, 1]  # L2 regularization term on weight
}

# Initialize RandomSearch with the model, parameter grid, and other parameters
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100, scoring='roc_auc', n_jobs=-1, cv=10, verbose=3)

# Perform RandomSearch on the training data
random_search.fit(data_train, labels_train)

# Print the best parameters found by RandomSearch
print("Best parameters found: ", random_search.best_params_)

# Train the model with early stopping using the best found parameters
model = random_search.best_estimator_
eval_set = [(data_test, labels_test)]  # Validation set for early stopping
model.fit(data_train, labels_train, eval_metric="logloss", eval_set=eval_set)

# Make predictions on the test set
preds_prob = model.predict_proba(data_test)

# Define a function to find the optimal threshold for classification
def find_optimal_threshold(predictions, y_test):
    min_sum = float('inf')
    optimal_threshold = 0.5

    # Iterate over possible thresholds from 0.3 to 0.8
    for threshold in np.arange(0.3, 0.8, 0.001):
        # Apply threshold to the second column of predictions
        preds = (np.array(predictions)[:,1] > threshold).astype(int)

        # Compute confusion matrix for the current threshold
        cm = confusion_matrix(y_test, preds)

        # Compute sum of off-diagonal elements
        off_diagonal_sum = 164 - np.trace(cm)

        # Update optimal threshold if this threshold is better
        if off_diagonal_sum < min_sum and cm[1][1]/np.sum(cm[1]) >0.5:
            min_sum = off_diagonal_sum
            optimal_threshold = threshold

    # Return the optimal threshold
    return optimal_threshold

# Find the optimal threshold for the test set predictions
threshold = find_optimal_threshold(preds_prob, labels_test)

# Uncomment the next few lines to print results for validation dataset
'''
# Apply the optimal threshold to the second column of the test set predictions
preds = (preds_prob[:, 1]> threshold).astype(int)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(labels_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Calculate and print the ROC AUC score
print("ROC AUC score:", roc_auc_score(labels_test, preds))

# Calculate and print the confusion matrix
print("Confusion matrix:\n", confusion_matrix(labels_test, preds))

# Calculate precision, recall, F1 score
precision, recall, fscore, _ = score(labels_test, preds)
print("Precision, Recall, Fscore:", precision, recall, fscore)
'''


# Make predictions on the original test set
pred_test = model.predict_proba(X_test)

# Apply the optimal threshold to the second column of the original test set predictions
preds = (pred_test[:, 1]> threshold).astype(int)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Calculate and print the ROC AUC score
print("ROC AUC score:", roc_auc_score(y_test, preds))

# Calculate and print the confusion matrix
print("Confusion matrix:\n", confusion_matrix(y_test, preds))

# Calculate precision, recall, F1 score
precision, recall, fscore, _ = score(y_test, preds)
print("Precision, Recall, Fscore:", precision, recall, fscore)