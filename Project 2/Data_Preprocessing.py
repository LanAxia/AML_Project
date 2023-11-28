# import library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

# sklearn
import sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV, SelectKBest, r_regression, f_regression
from sklearn.gaussian_process.kernels import Matern, RBF, CompoundKernel, Product, Sum, ExpSineSquared, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, make_scorer, f1_score
from sklearn.decomposition import PCA

# boost algorithm
import xgboost as xgb
import catboost as cat
import lightgbm as lgb

# torch
import torch
from torch import nn
from torch.nn import Module, Linear, Dropout
from torch.nn.functional import tanh, softmax, mse_loss, relu, sigmoid, binary_cross_entropy, nll_loss
from torch.optim import Adam, SGD

# bio library
import biosppy
from biosppy import storage
from biosppy.signals import ecg

DATA_DIR = "Data"
RESULT_DIR = "Result"

def check_template_result(templates: np.ndarray) -> list:
    # 检查有没有一个template里有多个心跳的情况
    check_result = True
    error_num = 0
    error_ids = []
    for template_i, template in enumerate(templates):
        peak_threshold = np.max(template) * 0.7
        peak_region = np.array(np.where(template > peak_threshold))
        if np.max(peak_region) - np.min(peak_region) > 0.5 * template.shape[0]:
            error_num += 1
            error_ids.append(template_i)
    return error_ids

def get_ecg_info(X, X_len):
    ts_lst = []
    filtered_lst = []
    rpeaks_lst = []
    templates_ts_lst = []
    templates_lst = []
    heart_rate_ts_lst = []
    heart_rate_lst = []

    error_ids = []
    part_error_lst = []
    for i, (signal, sig_len) in enumerate(zip(X, X_len)):
        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal[:sig_len], sampling_rate=300., show=False)
        # check template
        # if check_ecg_result(templates) == False:
        #     error_ids.append(i)
        #     error_templates.append(templates)
        #     continue

        # template_error_ids = check_template_result(templates) # 以较轻松的方式处理ecg处理异常
        template_error_ids = [] # 以较轻松的方式处理ecg处理异常

        # delete error data
        rpeaks = np.delete(rpeaks, template_error_ids, axis=0)
        templates_ts = np.delete(templates_ts, template_error_ids, axis=0)
        templates = np.delete(templates, template_error_ids, axis=0)
        if len(templates) < 1:
            error_ids.append(i)
            continue

        if len(template_error_ids) > 0:
            part_error_lst.append(len(ts_lst))

        ts_lst.append(ts) # Signal time axis reference (seconds)
        filtered_lst.append(filtered) # Filtered ECG signal
        rpeaks_lst.append(rpeaks) # R-peak location indices
        templates_ts_lst.append(templates_ts) # Templates time axis reference
        templates_lst.append(templates) # Extracted heartbeat templates
        heart_rate_ts_lst.append(heart_rate_ts) # Heart rate time axis reference (seconds)
        heart_rate_lst.append(heart_rate) # Instantaneous heart rate (bpm)
    return ts_lst, filtered_lst, rpeaks_lst, templates_ts_lst, templates_lst, heart_rate_ts_lst, heart_rate_lst

# get max height
max_height = None
for templates in templates_lst:
    for template in templates:
        if max_height is None or np.max(template) > max_height:
            max_height = np.max(template)
def scaler(template: np.array):
    result = template / max_height
    return result

# get average templates
def get_average_templates(templates):
    templates = scaler(templates)
    avg_templates = templates.sum(axis=0) / templates.shape[0]
    return avg_templates

def get_PQRST(templates_lst):
    PQRST = []
    for templates_i, templates in enumerate(templates_lst):
        template_PQRST = {
            "P": [], 
            "Q": [], 
            "R": [], 
            "S": [], 
            "T": [], 
            "P_id": [], 
            "Q_id": [], 
            "R_id": [], 
            "S_id": [], 
            "T_id": [], 
            "QRS": [], 
            "PR": [], 
            "PQ": [], 
            "ST": [], 
            "QT": [], 
            "error_state": []
        }
        for template_i, template in enumerate(templates):
            (P, Q, R, S, T), (P_id, Q_id, R_id, S_id, T_id), (QRS, PR, PQ, ST, QT), error_state = get_PQRST_from_template(template)
            template_PQRST["P"].append(P)
            template_PQRST["Q"].append(Q)
            template_PQRST["R"].append(R)
            template_PQRST["S"].append(S)
            template_PQRST["T"].append(T)
            template_PQRST["P_id"].append(P_id)
            template_PQRST["Q_id"].append(Q_id)
            template_PQRST["R_id"].append(R_id)
            template_PQRST["S_id"].append(S_id)
            template_PQRST["T_id"].append(T_id)
            template_PQRST["QRS"].append(QRS)
            template_PQRST["PR"].append(PR)
            template_PQRST["PQ"].append(PQ)
            template_PQRST["ST"].append(ST)
            template_PQRST["QT"].append(QT)
            template_PQRST["error_state"].append(error_state)
        PQRST.append(template_PQRST)
    return PQRST

def get_valid_features(rpeaks, heart_rate, PQRST):
    # rpeak
    rpeak_mean = np.array([np.average(x) for x in rpeaks])
    rpeak_median = np.array([np.median(x) for x in rpeaks])
    rpeak_std = np.array([np.std(x) for x in rpeaks])
    rpeak_min = np.array([np.min(x) for x in rpeaks])
    rpeak_max = np.array([np.max(x) for x in rpeaks])

    # heart rate
    hr_mean = np.array([np.average(x) for x in heart_rate])
    hr_median = np.array([np.median(x) for x in heart_rate])
    hr_std = np.array([np.std(x) for x in heart_rate])
    hr_min = np.array([np.min(x) for x in heart_rate])
    hr_max = np.array([np.max(x) for x in heart_rate])

    # PQRST
    # P
    P_mean = np.array([np.mean(x["P"]) for x in PQRST])
    P_median = np.array([np.median(x["P"]) for x in PQRST])
    P_std = np.array([np.std(x["P"]) for x in PQRST])
    P_min = np.array([np.min(x["P"]) for x in PQRST])
    P_max = np.array([np.max(x["P"]) for x in PQRST])

    # Q
    Q_mean = np.array([np.mean(x["Q"]) for x in PQRST])
    Q_median = np.array([np.median(x["Q"]) for x in PQRST])
    Q_std = np.array([np.std(x["Q"]) for x in PQRST])
    Q_min = np.array([np.min(x["Q"]) for x in PQRST])
    Q_max = np.array([np.max(x["Q"]) for x in PQRST])

    # R
    R_mean = np.array([np.mean(x["R"]) for x in PQRST])
    R_median = np.array([np.median(x["R"]) for x in PQRST])
    R_std = np.array([np.std(x["R"]) for x in PQRST])
    R_min = np.array([np.min(x["R"]) for x in PQRST])
    R_max = np.array([np.max(x["R"]) for x in PQRST])

    # S
    S_mean = np.array([np.mean(x["S"]) for x in PQRST])
    S_median = np.array([np.median(x["S"]) for x in PQRST])
    S_std = np.array([np.std(x["S"]) for x in PQRST])
    S_min = np.array([np.min(x["S"]) for x in PQRST])
    S_max = np.array([np.max(x["S"]) for x in PQRST])

    # T
    T_mean = np.array([np.mean(x["T"]) for x in PQRST])
    T_median = np.array([np.median(x["T"]) for x in PQRST])
    T_std = np.array([np.std(x["T"]) for x in PQRST])
    T_min = np.array([np.min(x["T"]) for x in PQRST])
    T_max = np.array([np.max(x["T"]) for x in PQRST])

    # P_i
    P_id_mean = np.array([np.mean(x["P_id"]) for x in PQRST])
    P_id_median = np.array([np.median(x["P_id"]) for x in PQRST])
    P_id_std = np.array([np.std(x["P_id"]) for x in PQRST])
    P_id_min = np.array([np.min(x["P_id"]) for x in PQRST])
    P_id_max = np.array([np.max(x["P_id"]) for x in PQRST])

    # Q_i
    Q_id_mean = np.array([np.mean(x["Q_id"]) for x in PQRST])
    Q_id_median = np.array([np.median(x["Q_id"]) for x in PQRST])
    Q_id_std = np.array([np.std(x["Q_id"]) for x in PQRST])
    Q_id_min = np.array([np.min(x["Q_id"]) for x in PQRST])
    Q_id_max = np.array([np.max(x["Q_id"]) for x in PQRST])

    # R_i
    R_id_mean = np.array([np.mean(x["R_id"]) for x in PQRST])
    R_id_median = np.array([np.median(x["R_id"]) for x in PQRST])
    R_id_std = np.array([np.std(x["R_id"]) for x in PQRST])
    R_id_min = np.array([np.min(x["R_id"]) for x in PQRST])
    R_id_max = np.array([np.max(x["R_id"]) for x in PQRST])

    # S_i
    S_id_mean = np.array([np.mean(x["S_id"]) for x in PQRST])
    S_id_median = np.array([np.median(x["S_id"]) for x in PQRST])
    S_id_std = np.array([np.std(x["S_id"]) for x in PQRST])
    S_id_min = np.array([np.min(x["S_id"]) for x in PQRST])
    S_id_max = np.array([np.max(x["S_id"]) for x in PQRST])

    # T_i
    T_id_mean = np.array([np.mean(x["T_id"]) for x in PQRST])
    T_id_median = np.array([np.median(x["T_id"]) for x in PQRST])
    T_id_std = np.array([np.std(x["T_id"]) for x in PQRST])
    T_id_min = np.array([np.min(x["T_id"]) for x in PQRST])
    T_id_max = np.array([np.max(x["T_id"]) for x in PQRST])

    # QRS
    QRS_mean = np.array([np.mean(x["QRS"]) for x in PQRST])
    QRS_median = np.array([np.median(x["QRS"]) for x in PQRST])
    QRS_std = np.array([np.std(x["QRS"]) for x in PQRST])
    QRS_min = np.array([np.min(x["QRS"]) for x in PQRST])
    QRS_max = np.array([np.max(x["QRS"]) for x in PQRST])

    # PR
    PR_mean = np.array([np.mean(x["PR"]) for x in PQRST])
    PR_median = np.array([np.median(x["PR"]) for x in PQRST])
    PR_std = np.array([np.std(x["PR"]) for x in PQRST])
    PR_min = np.array([np.min(x["PR"]) for x in PQRST])
    PR_max = np.array([np.max(x["PR"]) for x in PQRST])

    # PQ
    PQ_mean = np.array([np.mean(x["PQ"]) for x in PQRST])
    PQ_median = np.array([np.median(x["PQ"]) for x in PQRST])
    PQ_std = np.array([np.std(x["PQ"]) for x in PQRST])
    PQ_min = np.array([np.min(x["PQ"]) for x in PQRST])
    PQ_max = np.array([np.max(x["PQ"]) for x in PQRST])

    # ST
    ST_mean = np.array([np.mean(x["ST"]) for x in PQRST])
    ST_median = np.array([np.median(x["ST"]) for x in PQRST])
    ST_std = np.array([np.std(x["ST"]) for x in PQRST])
    ST_min = np.array([np.min(x["ST"]) for x in PQRST])
    ST_max = np.array([np.max(x["ST"]) for x in PQRST])

    # QT
    QT_mean = np.array([np.mean(x["QT"]) for x in PQRST])
    QT_median = np.array([np.median(x["QT"]) for x in PQRST])
    QT_std = np.array([np.std(x["QT"]) for x in PQRST])
    QT_min = np.array([np.min(x["QT"]) for x in PQRST])
    QT_max = np.array([np.max(x["QT"]) for x in PQRST])

    # error_state
    error_count = np.array([np.sum(x["error_state"]) for x in PQRST])
    error_mean = np.array([np.mean(x["error_state"]) for x in PQRST])

# valid features
    valid_features = [
        # rpeak
        rpeak_mean, 
        rpeak_median, 
        rpeak_std, 
        rpeak_min, 
        rpeak_max, 
        # heart rate
        hr_mean, 
        hr_median, 
        hr_std, 
        hr_min, 
        hr_max, 
        # P
        P_mean, 
        P_median, 
        P_std, 
        P_min, 
        P_max, 
        # Q
        Q_mean, 
        Q_median, 
        Q_std, 
        Q_min, 
        Q_max, 
        # R
        R_mean, 
        R_median, 
        R_std, 
        R_min, 
        R_max, 
        # S
        S_mean, 
        S_median, 
        S_std, 
        S_min, 
        S_max, 
        # T
        T_mean, 
        T_median, 
        T_std, 
        T_min, 
        T_max, 
        # P_id
        P_id_mean, 
        P_id_median, 
        P_id_std, 
        P_id_min, 
        P_id_max, 
        # Q_id
        Q_id_mean, 
        Q_id_median, 
        Q_id_std, 
        Q_id_min, 
        Q_id_max, 
        # R_id
        R_id_mean, 
        R_id_median, 
        R_id_std, 
        R_id_min, 
        R_id_max, 
        # S_id
        S_id_mean, 
        S_id_median, 
        S_id_std, 
        S_id_min, 
        S_id_max, 
        # T_id
        T_id_mean, 
        T_id_median, 
        T_id_std, 
        T_id_min, 
        T_id_max, 
        # QRS
        QRS_mean, 
        QRS_median, 
        QRS_std, 
        QRS_min, 
        QRS_max, 
        # PR
        PR_mean, 
        PR_median, 
        PR_std, 
        PR_min, 
        PR_max, 
        # PQ
        PQ_mean, 
        PQ_median, 
        PQ_std, 
        PQ_min, 
        PQ_max, 
        # ST
        ST_mean, 
        ST_median, 
        ST_std, 
        ST_min, 
        ST_max, 
        # QT
        QT_mean, 
        QT_median, 
        QT_std, 
        QT_min, 
        QT_max, 
        # error state
        error_count, 
        error_mean, 
    ]
    return valid_features


if __name__ == "__main__":
    # 加载数据
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), header=0, index_col=0)
    X_test_df = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"), header=0, index_col=0)
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"), header=0, index_col=0)

    X_train = X_train_df.values
    X_test = X_test_df.values
    y_train = y_train_df.values.ravel()

    # 获取有效长度
    X_train_len = []
    for row in X_train:
        tail_id = np.where(np.isnan(row))[0]
        if tail_id.shape[0] > 0:
            X_train_len.append(tail_id[0])
        else:
            X_train_len.append(X_train.shape[1])

    X_test_len = []
    for row in X_test:
        tail_id = np.where(np.isnan(row))[0]
        if tail_id.shape[0] > 0:
            X_test_len.append(tail_id[0])
        else:
            X_test_len.append(X_test.shape[1])

    X_train_len, X_test_len = np.array(X_train_len), np.array(X_test_len)

    # 获取ecg信息
    ts_lst, filtered_lst, rpeaks_lst, templates_ts_lst, templates_lst, heart_rate_ts_lst, heart_rate_lst = get_ecg_info(X_train, X_train_len)
    ts_lst_test, filtered_lst_test, rpeaks_lst_test, templates_ts_lst_test, templates_lst_test, heart_rate_ts_lst_test, heart_rate_lst_test = get_ecg_info(X_test, X_test_len)

    # 对所有的templates进行缩放
    templates_lst = [scaler(templates) for templates in templates_lst]

    # 获取平均templates
    avg_templates_lst = [get_average_templates(templates) for templates in templates_lst]

    # 获取PQRST数据
    PQRST = get_PQRST(templates_lst)
    PQRST_test = get_PQRST(templates_lst_test)

    # 处理rpeak
    rpeaks_new = []
    for rpeaks in rpeaks_lst:
        rpeaks_iterval = []
        for i in range(1, rpeaks.shape[0]):
            rpeaks_iterval.append(rpeaks[i] - rpeaks[i - 1])
        rpeaks_iterval = np.array(rpeaks_iterval)
        rpeaks_new.append(rpeaks_iterval)

    rpeaks_new_test = []
    for rpeaks in rpeaks_lst_test:
        rpeaks_iterval = []
        for i in range(1, rpeaks.shape[0]):
            rpeaks_iterval.append(rpeaks[i] - rpeaks[i - 1])
        rpeaks_iterval = np.array(rpeaks_iterval)
        rpeaks_new_test.append(rpeaks_iterval)

    # 处理heart_rate
    heart_rate_new = []
    for heart_rate in heart_rate_lst:
        if heart_rate.shape[0] == 0:
            heart_rate = np.array(-100)
        heart_rate_new.append(heart_rate)

    heart_rate_new_test = []
    for heart_rate in heart_rate_lst_test:
        if heart_rate.shape[0] == 0:
            heart_rate = np.array(-100)
        heart_rate_new_test.append(heart_rate)

    valid_features = get_valid_features(rpeaks_new, heart_rate_new, PQRST)
    valid_features_test = get_valid_features(rpeaks_new_test, heart_rate_new_test, PQRST_test)

    # 生成新的训练数据 X_train_features
    X_train_features = []
    for feature in valid_features:
        feature = feature.reshape((X_train.shape[0], -1))
        X_train_features.append(feature)
    X_train_features = np.concatenate(X_train_features, axis=1)

    X_test_features = []
    for feature in valid_features_test:
        feature = feature.reshape((X_test.shape[0], -1))
        X_test_features.append(feature)
    X_test_features = np.concatenate(X_test_features, axis=1)

    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    # 保存数据
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_train_len.npy"), X_train_len)
    np.save(os.path.join(DATA_DIR, "X_test_len.npy"), X_test_len)
    