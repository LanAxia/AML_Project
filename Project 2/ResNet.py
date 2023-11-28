import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import biosppy.signals.ecg as ecg
import pandas as pd
import neurokit2 as nk
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


def slide_and_cut(tmp_data, tmp_label, tmp_pid):
    out_pid = []
    out_data = []
    out_label = []
    window_size = 6000
    tmp_stride = 500
    for i in range(tmp_data.shape[0]):
        tmp_ts = tmp_data[i, :]
        mask = np.isnan(tmp_ts)
        tmp_ts = tmp_ts[~mask]
        for j in range(0, len(tmp_ts) - window_size, tmp_stride):
            out_pid.append(tmp_pid[i])
            out_data.append(tmp_ts[j:j + window_size])
            out_label.append(tmp_label[i])
    out_data = np.array(out_data, dtype=np.float32)
    out_pid = np.array(out_pid)
    out_label = np.expand_dims(np.array(out_label), axis=1)
    return out_data, out_label, out_pid


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        out = self.relu(out)
        return out


class MyModel(nn.Module):
    def __init__(self, n_dim, n_split):
        super(MyModel, self).__init__()
        self.n_dim = n_dim
        self.n_split = n_split

        self.conv1 = nn.Conv1d(1, 32, kernel_size=8, stride=4, padding=2)
        # self.mp1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=5, padding=2)
        # self.mp2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.res_blocks = nn.Sequential(
            ResidualBottleneckBlock(64, 64, stride=1),
            ResidualBottleneckBlock(256, 256, stride=1),
            ResidualBottleneckBlock(1024, 1024, stride=1)
        )

        self.conv3 = nn.Conv1d(4096, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(300, 1, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(1)

        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, self.n_dim)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # x = x.view(-1, self.n_split, 64)
        x = self.res_blocks(x)
        x = self.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        x = x.view(-1, self.n_split, 512)
        x, _ = self.lstm(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        feature_layer = F.sigmoid(self.fc1(x))
        x = self.dropout(feature_layer)
        x = F.softmax(self.fc2(x), dim=2)
        return feature_layer, x


# Instantiate the model
n_dim = 6000
n_split = 300
model = MyModel(n_dim, n_split)

data_X = pd.read_csv("./X_train.csv", header=0, index_col=0)
data_y = pd.read_csv("./y_train.csv", header=0, index_col=0).to_numpy()

# data_X = data_X.iloc[0:100]
# data_y = data_y[0:100, :]

data = []
for i in range(data_X.shape[0]):
    tem = data_X.iloc[i].dropna().to_numpy()
    ecg_fixed, is_inverted = nk.ecg_invert(tem, sampling_rate=300, show=False)
    filtered = ecg.ecg(ecg_fixed, sampling_rate=300, show=False)['filtered']
    data.append(filtered)
max_len = max(row.shape[0] for row in data)
new_data = np.array([np.pad(row, (0, max_len - row.shape[0]), constant_values=np.nan) for row in data])

X_train, X_test, y_train, y_test = train_test_split(new_data, data_y, test_size=0.2, random_state=42)

train_data, train_label, train_pid = slide_and_cut(X_train, y_train, np.arange(X_train.shape[0]))

train_data = torch.from_numpy(train_data)
train_label = torch.from_numpy(train_label)

dataset_train = TensorDataset(train_data, train_label)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 50
for epoch in range(epochs):
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    for i, data in enumerate(dataloader_train, 0):
        inputs, labels = data
        optimizer.zero_grad()

        labels = np.squeeze(labels)
        target = torch.squeeze(F.one_hot(labels, num_classes=4)).float()
        feature_layer, outputs = model(inputs)
        outputs = outputs.view(-1, 4)
        _, predicted = torch.max(outputs, 1)

        loss = criterion(outputs, target)
        total_correct += (predicted == labels).sum().item()
        total_samples += inputs.size(0)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_predictions.extend(predicted.numpy())
        all_targets.extend(labels.numpy())

    all_predictions = np.array(all_predictions).reshape((-1, 1))
    all_targets = np.array(all_targets).reshape((-1, 1))
    avg_acc = total_correct / total_samples
    overall_f1 = f1_score(all_targets, all_predictions, average='macro')
    cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2, 3])
    print(f'TRAINING:Epoch {epoch + 1}/{10}, Loss: {running_loss:.4f}, Accuracy: {avg_acc:.4f}, F1: {overall_f1:.4f}')
    print(cm)

# Save the trained model
torch.save(model.state_dict(), './best_model.pth')

test_data, test_label, test_pid = slide_and_cut(X_test, y_test,
                                                np.arange(X_train.shape[0], X_train.shape[0] + X_test.shape[0]))
test_data = torch.tensor(test_data, dtype=torch.float32)
num_of_test = len(test_data)

tmp_feature = []
tmp_predict = []

for i in range(num_of_test):
    tmp_testX = torch.tensor(test_data[i], dtype=torch.float32)
    feature_layer, prediction = model(tmp_testX)
    tmp_feature.extend(feature_layer.detach().numpy())
    tmp_predict.extend(prediction.detach().numpy())

tmp_feature = np.array(tmp_feature)
tmp_predict = np.array(tmp_predict)
new_feature = []

for i in range(X_train.shape[0], X_train.shape[0] + X_test.shape[0]):
    mask = np.where(test_pid == i)
    pid_feature = tmp_feature[mask]
    pid_predict = tmp_predict[mask]
    counts = np.bincount(np.argmax(pid_predict, axis=1))
    classes = np.argmax(pid_predict, axis=1)
    weights = np.max(pid_predict, axis=1) * counts[classes]
    weighted_features = np.average(pid_feature, axis=0, weights=weights)
    new_feature.append(weighted_features)
new_feature = np.array(new_feature)

# test_pid = np.array(test_pid, dtype=np.string_)
#
# y_num = len(test_pid)
# features = [[0. for j in range(32)] for i in range(y_num)]
# y_pre = [[0. for j in range(4)] for i in range(y_num)]
# y_sec_pre = [[0. for j in range(4)] for i in range(y_num)]
# y_third_pre = [[0. for j in range(4)] for i in range(y_num)]
#
# for j in range(len(tmp_feature)):
#     feature_pred = np.array(tmp_feature[j], dtype=np.float32)
#     i_pred = np.array(pre[j], dtype=np.float32)
#     cur_pid = str(test_pid[j], 'utf-8')
#
#     list_id = pid_map[cur_pid]
#     temp_feature = np.array(features[list_id], dtype=np.float32)
#     temp_pre = np.array(y_pre[list_id], dtype=np.float32)
#     temp_sec_pre = np.array(y_sec_pre[list_id], dtype=np.float32)
#     temp_third_pre = np.array(y_third_pre[list_id], dtype=np.float32)
#
#     max_p = temp_pre[np.argmax(temp_pre)]
#     max_sec_p = temp_sec_pre[np.argmax(temp_sec_pre)]
#     max_third_p = temp_third_pre[np.argmax(temp_third_pre)]
#     sec_p = 0
#     sec_sec_p = 0
#     sec_third_p = 0
#
#     for k in range(len(temp_pre)):
#         if temp_pre[k] == max_p:
#             continue
#         if temp_pre[k] > sec_p:
#             sec_p = temp_pre[k]
#
#         if temp_sec_pre[k] == max_sec_p:
#             continue
#         if temp_sec_pre[k] > sec_sec_p:
#             sec_sec_p = temp_sec_pre[k]
#
#         if temp_third_pre[k] == max_third_p:
#             continue
#         if temp_third_pre[k] > sec_third_p:
#             sec_third_p = temp_third_pre[k]
#
#     cur_max_p = i_pred[np.argmax(i_pred)]
#     cur_sec_p = 0
#
#     for k in range(len(i_pred)):
#         if i_pred[k] == cur_max_p:
#             continue
#         if i_pred[k] > cur_sec_p:
#             cur_sec_p = i_pred[k]
#
#     if (cur_max_p - cur_sec_p) > (max_p - sec_p):
#         y_third_pre[list_id] = y_sec_pre[list_id]
#         y_sec_pre[list_id] = y_pre[list_id]
#         y_pre[list_id] = i_pred
#     elif (cur_max_p - cur_sec_p) > (max_sec_p - sec_sec_p):
#         y_third_pre[list_id] = y_sec_pre[list_id]
#         y_sec_pre[list_id] = i_pred
#     elif (cur_max_p - cur_sec_p) > (max_third_p - sec_third_p):
#         y_third_pre[list_id] = i_pred
#
#     max_f = 0
#     for k in range(len(temp_feature)):
#         if temp_feature[k] > max_f:
#             max_f = temp_feature[k]
#     if max_f > 0:
#         feature_pred = (feature_pred + temp_feature) / 2
#
#     features[list_id] = feature_pred

# 输出结果
out_feature = []
for i in range(len(features)):
    out_feature.append(features[i])

out_feature = np.array(out_feature)
