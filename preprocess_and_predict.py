import sys
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import math
import scipy
from scipy.signal import find_peaks
sys.path.append(os.path.abspath("../../Fitness Tracker Project/data-science-template-main/src/features"))
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema

exercise_to_cutoff = {
    'bench': 0.4,
    'squat': 0.35,
    'ohp' : 0.35,
    'row': 0.65,
    'dead': 0.4
}

exercise_to_column = {
    'bench': 'acc_r',
    'squat': 'acc_r',
    'ohp' : 'acc_r',
    'row': 'gyr_r',
    'dead': 'acc_r'
}
exercise_names = {
    'bench': 'Bench Press',
    'squat': 'Squat',
    'ohp': 'Over Head Press',
    'dead': 'Deadlift',
    'row': 'Row',
}
acc_file_path= '../../Fitness Tracker Project/data-science-template-main/data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv'
gyr_file_path= '../../Fitness Tracker Project/data-science-template-main/data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv'
def read_data_from_files(acc_file_path, gyr_file_path):
    # Read accelerometer data
    acc_df = pd.read_csv(acc_file_path)
    # Read gyroscope data
    gyr_df = pd.read_csv(gyr_file_path)

    # Extract features from file name
    """participant = acc_file_path.split("/")[-1].split("-")[0]
    label = acc_file_path.split("/")[-1].split("-")[1]
    category = acc_file_path.split("/")[-1].split("-")[2].rstrip("123").rstrip("_MetaWear_2019")"""

    acc_set = 1
    gyr_set = 1
    # Add participant, label, and category to both dataframes
    acc_df["participant"] = "Vedant"
    acc_df["label"] = "label"
    acc_df["category"] = "category"
    acc_df["set"] = acc_set
    gyr_df["participant"] = "Vedant"
    gyr_df["label"] = "label"
    gyr_df["category"] = "category"
    gyr_df["set"] = gyr_set
   
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit='ms')

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
   
    return acc_df, gyr_df

def merge_data(acc_df, gyr_df):
    data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
    data_merged.columns = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "participant",
        "label",
        "category",
        "set"
    ]
    return data_merged

def resample_data(data_merged):
    sampling = {
        "acc_x": "mean",
        'acc_y': "mean",
        'acc_z': "mean",
        'gyr_x': "mean", 
        'gyr_y': "mean", 
        'gyr_z': "mean", 
        'participant': "last",
        'label': "last", 
        'category': "last", 
        'set': "last"
    }

    days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
    data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
    data_resampled["set"] = data_resampled["set"].astype("int")
   
    return data_resampled

def mark_outliers_chauvenet(dataset, col, C=2):

    #dataset = data_resampled
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

def remove_outliers(df, outlier_columns):
    outliers_removed_df = df.copy()

    for col in outlier_columns:
        for label in df["label"].unique():
            dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
            dataset.loc[dataset[col + "_outlier"], col] = np.nan
            # Update the col in original df
            outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[col]
            n_outliers = len(dataset) - len(dataset[col].dropna())
            print(f"Removed {n_outliers} from {col} for {label}")

    return outliers_removed_df

def build_features(df):
    predictor_columns = df.columns[:6].to_list()

    for col in predictor_columns:
        df[col] = df[col].interpolate()
    
    # Calculating set duration
    for s in df["set"].unique():
        stop = df[df["set"] == s].index[-1]
        start = df[df["set"] == s].index[0]
        duration = stop - start
        df.loc[df["set"] == s, "duration"] = duration.seconds

    # Butterworth lowpass filter
    LowPass = LowPassFilter()
    fs = 1000 / 200
    cutoff = 1.3
    for col in predictor_columns:
        df = LowPass.low_pass_filter(data_table=df, col=col, sampling_frequency=fs, cutoff_frequency=cutoff, order=5)
        df[col] = df[col + "_lowpass"]
        del df[col + "_lowpass"]

    # Principal component analysis PCA
    PCA = PrincipalComponentAnalysis()
    df = PCA.apply_pca(df, predictor_columns, 3)

    # Sum of Squared attributes
    df["acc_r"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["gyr_r"] = np.sqrt(df["gyr_x"]**2 + df["gyr_y"]**2 + df["gyr_z"]**2)

    # Temporal Abstraction
    NumAbs = NumericalAbstraction()
    predictor_columns += ["acc_r", "gyr_r"]
    ws = int(1000 / 200)

    df_list = []
    for s in df["set"].unique():
        subset = df[df["set"] == s].copy()
        for col in predictor_columns:
            subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
            subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
        df_list.append(subset)

    df = pd.concat(df_list)

    # Frequency features
    df_freq = df.copy().reset_index()
    FreqAbs = FourierTransformation()
    fs = int(1000 / 200)
    ws = int(2000 / 200)

    df_freq_list = []
    for s in df_freq["set"].unique():
        subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
        subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
        df_freq_list.append(subset)

    df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
    df_freq = df_freq.dropna()
    df_freq = df_freq.iloc[::2]

    # Kmeans clustering
    cluster_columns = ["acc_x", "acc_y", "acc_z"]
    kmeans = KMeans(5, n_init=20, random_state=0)
    subset = df_freq[cluster_columns]
    df_freq["cluster"] = kmeans.fit_predict(subset)

    return df_freq

def preprocess_data(acc_file_path, gyr_file_path):
    acc_df, gyr_df = read_data_from_files(acc_file_path, gyr_file_path)
    data_merged = merge_data(acc_df, gyr_df)
    data_resampled = resample_data(data_merged)
    outlier_columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    data_no_outliers = remove_outliers(data_resampled, outlier_columns)
    data_with_features = build_features(data_no_outliers)
    return data_with_features

def preprocess_data_count_reps(acc_file_path, gyr_file_path):
    acc_df, gyr_df = read_data_from_files(acc_file_path, gyr_file_path)
    data_merged = merge_data(acc_df, gyr_df)
    resampled_count_reps = resample_data(data_merged)
    acc_r = resampled_count_reps["acc_x"] **2 + resampled_count_reps["acc_y"] **2 + resampled_count_reps["acc_z"] **2
    gyr_r = resampled_count_reps["gyr_x"] **2 + resampled_count_reps["gyr_y"] **2 + resampled_count_reps["gyr_z"] **2

    resampled_count_reps["acc_r"] = np.sqrt(acc_r)
    resampled_count_reps["gyr_r"] = np.sqrt(gyr_r)
    return resampled_count_reps

def count_reps(dataset, exercise):
    cutoff = exercise_to_cutoff.get(exercise, 0.4) # Default to 0.4 if not found
    col = exercise_to_column.get(exercise, "acc_r")
    exercise_name = exercise_names.get(exercise, "Unknown")

    # Apply lowpass filter with the determined cutoff frequency
    LowPass = LowPassFilter()
    fs = 1000 / 200
    data_filtered = LowPass.low_pass_filter(data_table=dataset, col=col, sampling_frequency=fs, cutoff_frequency=cutoff, order=10)

    # Find peaks in the filtered data to count repetitions
    indexes = argrelextrema(data_filtered[col+"_lowpass"].values, np.greater)
    peaks = data_filtered.iloc[indexes]
    # Plot the data and peaks
    fig, ax = plt.subplots()
    plt.plot(data_filtered[f"{col}_lowpass"])
    plt.plot(peaks[f"{col}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{col}_lowpass")
    plt.title(f"{exercise_name}: {len(peaks)} Reps")
    plt.show()
    
    # Save plot to static folder
    plot_filename = '../../Fitness Tracker Project/data-science-template-main/static/reps_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    
    return len(peaks), plot_filename


def load_model(model_path='models/exercise_model.pkl'):
    return joblib.load(model_path)

def predict(model, data):
    model_training_data = pd.read_csv("../../Fitness Tracker Project/data-science-template-main/data/external/model_training_data.csv")
    required_columns = model_training_data.columns.tolist()
    
    
    inter_data = data.drop(["participant", "category", "set"], axis=1)
    
    
    basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    square_features = ["acc_r", "gyr_r"]
    pca_features = ["pca_1", "pca_2", "pca_3"]
    time_features = [f for f in inter_data.columns if "_temp_" in f]
    frequency_features = [f for f in inter_data.columns if ("_freq" in f) or ("_pse_" in f)]
    cluster_features = ["cluster"]
    
    feature_set_1 = basic_features
    feature_set_2 = list(set(basic_features + square_features + pca_features))
    feature_set_3 = list(set(feature_set_2 + time_features))
    feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))

    data_model = data[feature_set_4]
    
    #data_model = data_model.drop("epoch (ms)", axis= 1)
    for col in required_columns:
        if col not in data_model.columns:
            data_model[col] = 0

    data_model = data_model[required_columns]
    data_model.drop(columns=['epoch (ms)'], inplace=True)

    predictions = model.predict(data_model)
    prediction_probabilities = model.predict_proba(data_model)
    accuracy = model.score(data_model, predictions)
    return predictions, prediction_probabilities, accuracy

def main(acc_file_path, gyr_file_path, model_path='../../Fitness Tracker Project/data-science-template-main/src/models/exercise_model.pkl'):
    data = preprocess_data(acc_file_path, gyr_file_path)
    model = load_model(model_path)
    predictions, prediction_probabilities, accuracy = predict(model, data)
    resampled_count_reps = preprocess_data_count_reps(acc_file_path, gyr_file_path)
    reps, plot_filename = count_reps(resampled_count_reps, predictions[0])
    return predictions, prediction_probabilities, accuracy, reps, plot_filename
