import numpy as np
from scipy import signal

from sklearn.metrics import f1_score
import pandas as pd

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
import torch.nn.functional as F

import neurokit2 as nk
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg


def fourier(data):
    spectrogram_nperseg = 64
    spectrogram_noverlap = 32
    data = np.nan_to_num(data)
    _, _, Sxx = spectrogram(data, nperseg=spectrogram_nperseg, noverlap=spectrogram_noverlap)
    return Sxx


def spectrogram(data, nperseg=64, noverlap=32, log_spectrogram=True):
    fs = 300
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx, [0, 2, 1])
    if log_spectrogram:
        Sxx = abs(Sxx)
        mask = Sxx > 0
        Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx


class F1Loss(torch.nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        tp = torch.sum(y_true * y_pred, dim=0)
        fp = torch.sum((1 - y_true) * y_pred, dim=0)
        fn = torch.sum(y_true * (1 - y_pred), dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)

        return 1 - f1.mean()


class FourierModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, "same"),
            nn.MaxPool2d(3, 3),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, "same"),
            nn.MaxPool2d(3, 3),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, "same"),
            nn.MaxPool2d(3, 3),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(640, 256),
            nn.Linear(256, 256),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(256, 4),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer_1(x.float())
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        output = self.output_layer(x)
        return output


class FourierData(Dataset):
    def __init__(self, data_X, data_y):
        self.data = data_X
        self.target = data_y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.target[idx]
        return data, target


data_X = pd.read_csv("./X_train.csv", header=0, index_col=0)
data_y = pd.read_csv("./y_train.csv", header=0, index_col=0).to_numpy()

data = []
for i in range(data_X.shape[0]):
    tem = data_X.iloc[i].dropna().to_numpy()
    ecg_fixed, is_inverted = nk.ecg_invert(tem, sampling_rate=300, show=False)
    filtered = ecg.ecg(ecg_fixed, sampling_rate=300, show=False)['filtered']
    data.append(filtered)
max_len = max(row.shape[0] for row in data)
new_data = np.array([np.pad(row, (0, max_len - row.shape[0]), constant_values=np.nan) for row in data])

data_X = np.expand_dims(fourier(new_data), axis=1)
print(data_X.shape)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
dataset_train = FourierData(X_train, y_train)
dataset_test = FourierData(X_test, y_test)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

model = FourierModel()
criterion = nn.BCEWithLogitsLoss()
# criterion = F1Loss()
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    for batch_data, batch_target in dataloader_train:
        labels = torch.squeeze(batch_target)
        batch_target = torch.squeeze(F.one_hot(batch_target, num_classes=4)).float()
        output = model(batch_data)
        loss = criterion(output, batch_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_data.size(0)
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += batch_data.size(0)

        all_predictions.extend(predicted.numpy())
        all_targets.extend(labels.numpy())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    overall_f1 = f1_score(all_targets, all_predictions, average='macro')

    print(f'TRAINING:Epoch {epoch + 1}/{1000}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, F1: {overall_f1:.4f}')

    if (epoch+1) % 10 == 0:
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch_data, batch_target in dataloader_test:
                labels = torch.squeeze(batch_target)
                batch_target = torch.squeeze(F.one_hot(batch_target, num_classes=4)).float()
                output = model(batch_data)
                loss = criterion(output, batch_target)

                total_loss += loss.item() * batch_data.size(0)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_data.size(0)

                all_predictions.extend(predicted.numpy())
                all_targets.extend(labels.numpy())

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            overall_f1 = f1_score(all_targets, all_predictions, average='macro')

            print(f'VALIDATION:Epoch {epoch + 1}/{1000}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, F1: {overall_f1:.4f}')