import os
import glob
import pandas as pd
import random

# Path to your EV timeseries directory
folder_path = "../data_ev/EV_1year/"

ev_battery_sizes_kwh = [
    55.0, #https://en.wikipedia.org/wiki/Ford_Explorer_EV
    82.0, #https://en.wikipedia.org/wiki/Ford_Explorer_EV
    42.0, #https://en.wikipedia.org/wiki/Renault_4_E-Tech
    101.7 #bmw i7 xDrive60 https://configure.bmw.be/fr_BE/configure/G70E/51EJ/FVCJL,P0C4A,S01FF,S0230,S02PA,S02VB,S02VC,S02VH,S02VS,S0302,S0337,S033E,S03DM,S03DN,S0407,S0416,S0428,S043C,S0454,S0465,S046A,S04A2,S04F5,S04FM,S04GQ,S04HA,S04T2,S04T6,S04U6,S04U9,S04V1,S0548,S05AU,S05AW,S05DW,S0654,S06AE,S06AF,S06AK,S06C4,S06FR,S06NX,S06PA,S06U3,S06U7,S0710,S0760,S0775,S07CG,S07R7,S0851,S0886,S08R3,S08R9,S08TF,S08WN,S09DA,S09T1,S09T2,S0ZEV,S0ZPE,S0ZT7,S0ZTA/SE000001
]

# Get all CSV files starting with EV_
file_list = glob.glob(os.path.join(folder_path, "EV_*.csv"))

for file_path in file_list:
    try:
        df = pd.read_csv(file_path)

        # Convert first column to datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

        # Change year from 2019 to 2023
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda dt: dt.replace(year=2023) if dt.year == 2019 else dt)

        # Pick random battery size
        battery_size = random.choice(ev_battery_sizes_kwh)

        # Rename file
        file_name = os.path.basename(file_path)
        base_name = file_name.split('.')[0]
        new_file_name = f"{base_name}_{int(battery_size)}.csv"
        new_file_path = os.path.join(folder_path, new_file_name)

        # Save updated CSV
        df.to_csv(new_file_path, index=False)

        # Optionally remove the old file
        os.remove(file_path)

        print(f"Renamed and updated: {file_name} -> {new_file_name}")

    except Exception as e:
        print(f"Failed for {file_path}: {e}")
