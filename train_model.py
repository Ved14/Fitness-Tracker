import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# Train test split

df_train = df.drop(["participant", "category", "set"], axis=1)

X = df_train.drop(["label"],axis = 1)
y = df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42, stratify=y)

fig, ax = plt.subplots(figsize=(10,5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Total")
y_train.value_counts().plot(kind="bar", color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", color="royalblue", label="Test")
plt.legend()
plt.show()


# Splitting feature subsets

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
frequency_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse_" in f)]
cluster_features = ["cluster"]

print("Basic features:", len(basic_features))
print("square_features:", len(square_features))
print("pca_features:", len(pca_features))
print("time_features:", len(time_features))
print("frequency_features:", len(frequency_features))
print("cluster_features:", len(cluster_features))


feature_set_1 = basic_features
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))


# Forward feature selection using decision tree

learner = ClassificationAlgorithms()


selected_features = [
 'pca_1',
 'duration',
 'acc_x_freq_0.0_Hz_ws_10',
 'acc_y_freq_0.0_Hz_ws_10',
 'gyr_z_freq_weighted',
 'gyr_y_freq_1.5_Hz_ws_10',
 'acc_x_temp_mean_ws_5',
 'acc_x_freq_2.0_Hz_ws_10',
 'gyr_x_freq_0.0_Hz_ws_10',
 'acc_y_pse']
max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(max_features, X_train, y_train)

'''plt.plot(np.array(1, max_features +1 ,1), ordered_scores)
plt.xticks(np.array(1, max_features + 1, 1))
plt.show()'''


# Grid search for the best hyperparameters amd model selection

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features
]


feature_names = [
    "feature_set_1",
    "feature_set_2",
    "feature_set_3",
    "feature_set_4",
    "selected_features"
]


iterations = 1
score_df = pd.DataFrame()


for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])
    
    
    # creaing a grouped bar plot to compare the results
    
    score_df.sort_values(by="accuracy", ascending=False, inplace=True)
    
    plt.figure(figsize=(10,10))
    sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.7,1)
    plt.legend(loc="lower right")
    plt.show()
    
    # Selecting the best model and evaluating results

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y, = learner.random_forest(
            X_train[feature_set_4],
            y_train,
            X_test[feature_set_4],
            gridsearch=True,
        )


accuracy = accuracy_score(y_test, class_test_y)



classes = class_test_prob_y.columns
cm =  confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()



# Selecting the train & test data based on participants


participant_df = df.drop(["set", "category"], axis = 1)

X_train = participant_df[participant_df["participant"]!= "A"].drop("label", axis = 1)
X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis = 1)
y_train = participant_df[participant_df["participant"]!= "A"]["label"]
y_test = participant_df[participant_df["participant"]== "A"]["label"]


fig, ax = plt.subplots(figsize=(10,5))
df_train["label"].value_counts().plot(kind="bar", ax=ax, color="lightblue", label="Total")
y_train.value_counts().plot(kind="bar", color="dodgerblue", label="Train")
y_test.value_counts().plot(kind="bar", color="royalblue", label="Test")
plt.legend()
plt.show()

# Update the best model

len(feature_set_4)

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y, = learner.random_forest(
            X_train[feature_set_4],
            y_train,
            X_test[feature_set_4],
            gridsearch=True,
        )
X_train[feature_set_4].to_csv("../../data/external/model_training_data.csv")
# Save the trained Random Forest model
exercise_model = learner.get_model()
joblib.dump(exercise_model, 'exercise_model.pkl')

accuracy = accuracy_score(y_test, class_test_y)



classes = class_test_prob_y.columns
cm =  confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# Trying a more complex model

class_train_y, class_test_y, class_train_prob_y, class_test_prob_y, = learner.feedforward_neural_network(
            X_train[feature_set_4],
            y_train,
            X_test[feature_set_4],
            gridsearch=True,
        )

accuracy = accuracy_score(y_test, class_test_y)



classes = class_test_prob_y.columns
cm =  confusion_matrix(y_test, class_test_y, labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()