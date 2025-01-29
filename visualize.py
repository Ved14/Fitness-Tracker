import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl


# Loading data

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

#Plotting single column

set_df = df[df["set"]==1]

plt.plot(set_df["acc_y"].reset_index(drop=True))

for label in df["label"].unique():
    subset = df[df["label"]==label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()
    

mpl.style.use("seaborn-deep")
mpl.rcParams["figure.figsize"]=(20,5)
mpl.rcParams["figure.dpi"]=100

# Comparing different metrics

category_df = df.query("label=='squat'").query("participant=='A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
plt.legend()

# Plotting multiple axis

label = "squat"
participant = "A"
all_axis_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
fig, ax = plt.subplots()
all_axis_df[["acc_x","acc_y","acc_z"]].plot()
plt.legend()

# Plotting all combinations

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax)
            plt.title(f"{label}({participant})".title())
            plt.legend()


for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            plt.title(f"{label}({participant})".title())
            plt.legend()


# Combing the plots

label = "row"
participant = "A"
combined_plot_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,20))
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
plt.show()


# Combinging everything

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,20))
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
                        
            
            
            