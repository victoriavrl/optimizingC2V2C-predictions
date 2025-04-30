import pandas as pd
import glob
import os

folder_path = "../data/REC_data/"

# Find all Excel files ending with '2023.xlsx'
file_list = glob.glob(os.path.join(folder_path, "*2023.xlsx"))

combined_df = pd.DataFrame()

for file_path in file_list:
    try:
        # Read only the 'Export' sheet
        df = pd.read_excel(file_path, sheet_name='Export', parse_dates=True)

        # Ensure datetime is the index (first column assumed to be datetime)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        df = df.set_index(df.columns[0])
        df = df.sort_index()

        # Normalize column names
        original_columns = df.columns
        col_map = {col: col.strip().upper() for col in df.columns}
        reverse_col_map = {v: k for k, v in col_map.items()}
        df.rename(columns=col_map, inplace=True)

        # Find columns without 'TOTALE'
        injection_cols = [col for col in df.columns if "INJECTION" in col and "TOTALE" not in col]
        consommation_cols = [col for col in df.columns if "CONSOMMATION" in col and "TOTALE" not in col]

        print(f"\nUsing columns from {os.path.basename(file_path)}:")
        print("Injection columns:", injection_cols)
        print("Consommation columns:", consommation_cols)

        if injection_cols and consommation_cols:
            df['total_injection'] = df[injection_cols].sum(axis=1, skipna=True)
            df['total_consumption'] = df[consommation_cols].sum(axis=1, skipna=True)

            # Compute energy: positive if injection, negative if consumption
            df['energy'] = 0
            df.loc[df['total_injection'] > 0, 'energy'] = df['total_injection']
            df.loc[(df['total_injection'] == 0) & (df['total_consumption'] > 0), 'energy'] = -df['total_consumption']

            filtered = df[['total_injection', 'total_consumption', 'energy']].dropna(subset=['energy'])
            combined_df = pd.concat([combined_df, filtered])
        else:
            print(f"Missing required columns in {file_path}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Final cleanup and save
combined_df = combined_df.sort_index()
combined_df.index = combined_df.index.round('15min')
combined_df.to_csv("home/data.csv")
