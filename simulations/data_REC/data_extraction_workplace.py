import pandas as pd

# Load the Excel file
file_path = '../data/REC_data/GB_2024.xlsx'
df = pd.read_excel(file_path, sheet_name='2024')

# Convert and update datetime
df["FromDate"] = pd.to_datetime(df["FromDate (GMT+1)"])
df = df[~((df["FromDate"].dt.month == 2) & (df["FromDate"].dt.day == 29))]
df["datetime"] = df["FromDate"].apply(lambda dt: dt.replace(year=2023))

# Get relevant columns
residuelle_col = "0_Injection Résiduelle"
complementaire_cols = [col for col in df.columns if "volume complémentaire" in col.lower()]

# Compute the sum of Volume Complémentaire
df["sum_complementaire"] = df[complementaire_cols].sum(axis=1)


# Apply the logic for energy
def compute_energy(row):
    if row[residuelle_col] > 0:
        return row[residuelle_col]
    elif row["sum_complementaire"] > 0:
        return -row["sum_complementaire"]
    else:
        return 0


df["energy"] = df.apply(compute_energy, axis=1)

# Save result
output_df = df[["datetime", "energy"]]
output_df.to_csv("workplace/data.csv", index=False)
