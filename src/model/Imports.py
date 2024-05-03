import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import shuffle