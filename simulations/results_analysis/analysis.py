import pandas as pd
from glob import glob
import os

trip_data_files = glob('../data_ev/trips_data/*.csv')


def test_soc_below_20_reporting():
    results = []
    for path in trip_data_files:
        df = pd.read_csv(path, parse_dates=['datetime'])
        soc_below_20 = (df['SoC'] < 0.2).sum()
        total = len(df)
        percentage = round((soc_below_20 / total) * 100, 2) if total > 0 else 0.0
        results.append({
            'file': os.path.basename(path),
            'total_entries': total,
            'entries_below_0.2': soc_below_20,
            'percentage_below_0.2': percentage
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("results/soc_below_20_report.csv", index=False)


def test_soc_not_zero_or_negative(trip_data):
    soc_zero_or_negative = trip_data[trip_data['SoC'] <= 0]
    assert soc_zero_or_negative.empty, (
        f"SoC <= 0 found:\n{soc_zero_or_negative[['datetime', 'SoC']]}"
    )
