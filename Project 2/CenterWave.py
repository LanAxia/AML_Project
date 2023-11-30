import numpy as np
import pandas as pd
from scipy.signal import periodogram
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import neurokit2 as nk
from dtaidistance import dtw
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
import time
from fastdtw import fastdtw

# import os
#
#
# os.environ['OMP_NUM_THREADS'] = '1'
data_X = pd.read_csv("./X_train.csv", header=0, index_col=0)


# data_y = pd.read_csv("./y_train.csv", header=0, index_col=0).to_numpy()
# data = np.load("./feature_center_wave.npy")
# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(5, 2, i + 1)
#     plt.plot(np.arange(180), data[i, :])
#     plt.title(str(data_y[i]))
# plt.show()
# print(data.shape)
# assert 0

# data_y = pd.read_csv("./y_train.csv", header=0, index_col=0).to_numpy()

def calculate_dtw_distance(seq1, seq2):
    # Use fastdtw for efficient DTW calculation
    distance, _ = fastdtw(seq1, seq2)
    return distance


def calculate_all_dtw_distances(beats):
    n = len(beats)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist_matrix[i, j] = calculate_dtw_distance(beats[i], beats[j])
            dist_matrix[j, i] = dist_matrix[i, j]  # The distance matrix is symmetric
    return dist_matrix


center_waves = []
for i in range(data_X.shape[0]):
    tem = data_X.iloc[i].dropna().to_numpy()
    signal, is_inverted = nk.ecg_invert(tem, sampling_rate=300, show=False)
    r_peaks = ecg.hamilton_segmenter(signal, 300)['rpeaks']
    # beats = ecg.extract_heartbeats(signal, r_peaks, 300)['templates']
    beats = [tem[r_peaks[i]:r_peaks[i+1]] for i in range(len(r_peaks)-1)]
    print(i, beats.shape)
    if len(beats) != 0:
        dist_matrix = calculate_all_dtw_distances(beats)
        # dist_matrix = pairwise_distances(beats, metric=calculate_dtw_distance)
        similarity_matrix = np.exp(-dist_matrix ** 2 / (2. * np.var(dist_matrix)))
        clustering = SpectralClustering(n_clusters=4, affinity='precomputed')
        labels = clustering.fit_predict(similarity_matrix)

        biggest_cluster = np.argmax(np.bincount(labels))
        center_wave_index = np.argmin(dist_matrix[labels == biggest_cluster].sum(axis=0))
        center_wave = beats[center_wave_index]
        center_waves.append(center_wave)
    else:
        print("Error:" + str(i))
        center_waves.append(np.zeros((1, 180)))
center_waves = np.array(center_waves)
np.save("./feature_center_wave.npy", center_waves)
