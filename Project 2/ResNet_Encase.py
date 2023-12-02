import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


def general_data(data, label):
    out_data = []
    out_label = []
    window_size = 6000
    stride = 500

    for i in range(data.shape[0]):
        tem = data[i, :]
        tem = tem[~np.isnan(tem)]
        length = tem.shape[0]
        if length > window_size:
            for i in range(0, (length - window_size) // stride + 1):
                new_x_row = tem[i * stride: i * stride + window_size]
                out_data.append(new_x_row)
                out_label.append(label[i])
        else:
            new_data = np.zeros((window_size,))
            left = int(window_size / 2 - length // 2)
            right = left + length
            # print(left, right)
            new_data[left:right] = tem
            out_data.append(new_data)
            out_label.append(label[i])
    out_data = np.array(out_data, dtype=np.float32)
    out_label = np.array(out_label)
    return out_data, out_label


def score(model, x, y):
    model.eval()
    device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")
    with torch.no_grad():
        y_preds = []
        for i in range(0, x.shape[0], 64):
            x_batch = x[i:i + 64, :]
            x_batch = x_batch.reshape([x_batch.shape[0], 1, x_batch.shape[1]])
            _, y_pred = model(x_batch)
            y_pred = torch.argmax(torch.softmax(y_pred, dim=-1), dim=-1).to(cpu_device).detach().numpy()
            y_preds.append(y_pred)
        y_preds = np.concatenate(y_preds, axis=0)
        score = f1_score(y.to(cpu_device).detach().numpy(), y_preds, average="micro")
        cm = confusion_matrix(y_preds, y.to(cpu_device).detach().numpy(), labels=[0, 1, 2, 3])
        print("F1: " + str(score))
        print(cm)
    model.train()
    return score


class Residual_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, downsample=False, downsample_strides=2, kernel=3):
        super(Residual_Block, self).__init__()

        self.downsample = downsample
        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=kernel, stride=downsample_strides, padding=0),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
        )

        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=downsample_strides, padding=0),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)

        if self.downsample:
            residual = self.downsample_layer(x)

        out = out + residual
        out = self.relu(out)
        # print(out.shape)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channels=1, output_features=32, output_dim=4, dim=6000, n_split=300):
        super(ResNet, self).__init__()

        self.dim = dim
        self.n_split = n_split

        self.layer_1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer_2 = nn.Sequential(  # 8组残差视觉网络
            Residual_Block(64, 16, 64, True, 2),
            Residual_Block(64, 16, 64, True, 2),
            Residual_Block(64, 16, 128, True, 2),
            Residual_Block(128, 16, 128, True, 2),
            Residual_Block(128, 16, 256, True, 2),
            Residual_Block(256, 16, 256, True, 2),
            Residual_Block(256, 16, 512, True, 2, 1),
            Residual_Block(512, 16, 512, True, 2, 1),
        )

        self.layer_3 = nn.Sequential(  # BiLSTM
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True),
        )

        self.layer_4 = nn.Sequential(  # 解析为32维向量
            nn.Dropout(0.25),
            nn.Linear(512, output_features),
            nn.ReLU()
        )

        self.layer_5 = nn.Sequential(  # 解析为32维向量
            nn.Dropout(0.25),
            nn.Linear(output_features * 20, output_features),
            nn.Sigmoid()
        )

        self.layer_6 = nn.Sequential(  #
            nn.Dropout(0.25),
            nn.Linear(output_features, output_dim),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = x.view(-1, 1, self.n_split)
        # print(x.shape)
        output = self.layer_1(x)
        # print(output.shape)
        output = self.layer_2(output)
        # print(output.shape)
        output = output.view(-1, self.dim // self.n_split, 512)
        # print(output.shape)
        output, _ = self.layer_3(output)
        # print(output.shape)
        output = self.layer_4(output)
        # print(output.shape)
        output = output.view(-1, 32 * 20)
        # print(output.shape)
        feature = self.layer_5(output)
        # print(output.shape)
        output = self.layer_6(feature)
        # print(output.shape)
        return feature, output


if __name__ == "__main__":
    device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")

    data_X = pd.read_csv("./X_train.csv", header=0, index_col=0).to_numpy()
    data_y = pd.read_csv("./y_train.csv", header=0, index_col=0).to_numpy()
    data_X, data_y = general_data(data_X, data_y)
    print(data_X.shape)
    model = ResNet().to(device)

    class_sample_counts = np.bincount(data_y.squeeze())
    total_samples = len(data_y)
    class_weights = torch.tensor((1 / class_sample_counts) * (total_samples / 2), dtype=torch.float32).to(device)

    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(data_X)

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2)
    train_data = torch.from_numpy(X_train).to(device)
    train_label = torch.from_numpy(y_train).to(device)
    test_data = torch.from_numpy(X_test).to(device)
    test_label = torch.from_numpy(y_test).to(device)

    dataset_train = TensorDataset(train_data, train_label)
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 300
    best_valid = 0
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
            # target = torch.squeeze(F.one_hot(labels, num_classes=4)).float()

            feature, outputs = model(inputs.float())
            # outputs = outputs.view(-1, 4)
            _, predicted = torch.max(outputs, 1)

            # loss = criterion(outputs, target)
            loss = criterion(outputs, labels)
            total_correct += (predicted == labels).sum().item()
            total_samples += inputs.size(0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_predictions.extend(predicted.to(cpu_device).numpy())
            all_targets.extend(labels.to(cpu_device).numpy())

        all_predictions = np.array(all_predictions).reshape((-1, 1))
        all_targets = np.array(all_targets).reshape((-1, 1))
        avg_acc = total_correct / total_samples
        overall_f1 = f1_score(all_targets, all_predictions, average='micro')
        cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2, 3])
        print(
            f'TRAINING:Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Accuracy: {avg_acc:.4f}, F1: {overall_f1:.4f}')
        print(cm)
        if (epoch + 1) % 10 == 0:
            print(f'TRAINING:Epoch {epoch + 1}/{epochs}')
            f1 = score(model, test_data, test_label)
            if f1 > best_valid:
                best_valid = f1
                torch.save(model.state_dict(), "./Resnet_best.pt")
    torch.save(model.state_dict(), "./Resnet.pt")
