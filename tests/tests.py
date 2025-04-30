import pytest
import pandas as pd
import numpy as np
import os
from glob import glob
import re

trip_data_files = glob('../simulations/data_ev/trips_data/*.csv')


@pytest.fixture(params=trip_data_files, scope="module")
def trip_data(request):
    path = request.param
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.name = os.path.basename(path)
    return df


def extract_ev_info(ev_path):
    """Extract the EV name and battery capacity from the file path."""
    filename = os.path.basename(ev_path)
    match = re.match(r"(EV_\d+)_(\d+).*\.csv", filename)
    if match:
        ev_name = match.group(1)
        battery_capacity = int(match.group(2))
        return ev_name, battery_capacity
    else:
        raise ValueError(f"Invalid file format: {filename}")


def test_soc_consistency(trip_data):
    """Test if SoC is always the product of EbattCap and Ebattery."""
    df = trip_data.copy()
    ev_name, battery_capacity = extract_ev_info(trip_data.name)

    # Ensure SoC is always between 0 and 1
    df['SoC_check'] = np.isclose(df['SoC'], df['Ebattery'] / battery_capacity)

    false_rows = df[~df['SoC_check']]

    assert false_rows.empty, f"SoC mismatch found (should be Ebattery / EbattCap):\n{false_rows[['datetime', 'SoC', 'Ebattery']]}"


def test_ebattery_consistency(trip_data):
    df = trip_data.copy()
    df['Ebattery_check'] = np.isclose(df['EbattR'] + df['EbattG'], df['Ebattery'])
    false_rows = df[~df['Ebattery_check']]
    assert false_rows.empty, f"Ebattery mismatch found:\n{false_rows[['datetime', 'EbattR', 'EbattG', 'Ebattery']]}"


@pytest.mark.parametrize("col", ['EbattR', 'EbattG', 'Ebattery', 'consumption', 'Eneeded'])
def test_no_negative_energy_values(trip_data, col):
    data_to_check = trip_data.iloc[:-1]  # Exclude the last row bc EV can stop if the SoC is negative
    negative_rows = data_to_check[data_to_check[col] < 0]
    assert negative_rows.empty, f"Negative values found in column '{col}':\n{negative_rows[['datetime', col]]}"


def test_soc_within_bounds(trip_data):
    invalid_soc = trip_data[(trip_data['SoC'] < 0) | (trip_data['SoC'] > 1)]
    assert invalid_soc.empty, f"SoC out of [0,1] range:\n{invalid_soc[['datetime', 'SoC']]}"


def test_datetime_increments_by_15_minutes(trip_data):
    trip_data = trip_data.sort_values('datetime')
    trip_data['time_diff'] = trip_data['datetime'].diff().dt.total_seconds() / 60  # in minutes
    invalid_time_diff = trip_data[(~trip_data['time_diff'].isna()) & (trip_data['time_diff'] != 15)]
    assert invalid_time_diff.empty, f"Found rows with invalid time difference (not 15 minutes):\n{invalid_time_diff[['datetime', 'time_diff']]}"


def test_no_duplicate_datetimes(trip_data):
    duplicated = trip_data['datetime'].duplicated().sum()
    assert duplicated == 0, f"Found {duplicated} duplicated datetime entries."


def test_driving_state_energy_decrease(trip_data):
    """Test that during driving:
       - Ebattery = previous Ebattery - consumption
       - EbattG decreases compared to previous row (unless already 0)
    """
    df = trip_data.copy().sort_values('datetime').reset_index(drop=True)

    # Shift Ebattery and EbattG to get previous values
    df['Ebattery_prev'] = df['Ebattery'].shift(1)
    df['EbattG_prev'] = df['EbattG'].shift(1)

    driving_rows = df[(df['state'] == 'driving') & (df.index > 0)]

    # Check Ebattery consistency
    ebattery_mismatch = driving_rows[
        ~np.isclose(driving_rows['Ebattery'], driving_rows['Ebattery_prev'] - driving_rows['consumption'])
    ]

    # EbattG should decrease OR stay 0 if it was already 0
    ebattg_wrong = driving_rows[
        ~(
                (driving_rows['EbattG_prev'] == 0) & (driving_rows['EbattG'] == 0) |  # stays 0
                (driving_rows['EbattG'] < driving_rows['EbattG_prev'])  # decreases
        )
    ]

    assert ebattery_mismatch.empty, f"Mismatch in Ebattery during driving:\n{ebattery_mismatch[['datetime', 'Ebattery_prev', 'consumption', 'Ebattery']]}"
    assert ebattg_wrong.empty, f"EbattG did not decrease or stay at 0 during driving:\n{ebattg_wrong[['datetime', 'EbattG_prev', 'EbattG']]}"


def find_rec_charging_periods(df, rec_location):
    """Return a mask where EV is at REC and datetime is between plug-in and Plug_out_pred."""
    mask = pd.Series(False, index=df.index)

    df_sorted = df.sort_values('datetime').reset_index(drop=True)

    # Identify when the EV is at home or workplace
    rec_rows = df_sorted[df_sorted['state'] == rec_location]

    for idx in rec_rows.index:
        plug_in_time = df_sorted.loc[idx, 'datetime']
        plug_out_pred = df_sorted.loc[idx, 'Plug_out_pred']

        # If plug_out_pred is not NaT or NaN
        if pd.notna(plug_out_pred):
            # Mark all rows between plug-in and plug-out_pred
            mask |= (df_sorted['datetime'] >= plug_in_time) & (df_sorted['datetime'] < plug_out_pred)

    return mask


@pytest.mark.parametrize("rec_location", ["home", "workplace"])
def test_charging_behavior_with_rec_predictions(trip_data, rec_location):
    """Test that during plug-in time and predicted plug-out at REC:
       if REC predictions > 0 and SoC < 0.8, then EchargedBattery > 0,
       Ebattery increases, and EbattR increases.
    """
    if "noSM" in trip_data.name:
        pytest.skip("Skipping non-smart-charging trip data file.")

    df = trip_data.copy().sort_values('datetime').reset_index(drop=True)

    # Load REC prediction data
    rec_pred_path = f"../simulations/data_REC/{rec_location}/data.csv"
    rec_pred = pd.read_csv(rec_pred_path, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)

    # Merge trip data with REC predictions
    df = df.merge(rec_pred[['datetime', 'energy']], on='datetime', how='left')

    # Shift previous values
    df['Ebattery_prev'] = df['Ebattery'].shift(1)
    df['EbattR_prev'] = df['EbattR'].shift(1)

    inside_session = find_rec_charging_periods(df, rec_location)

    session_rows = df[inside_session]

    condition = (
            (session_rows['energy'] > 0) &
            (session_rows['SoC'] < 0.8) &
            (session_rows['state'] == rec_location)
    )

    rows_to_check = session_rows[condition]

    # Now check:
    echarged_issue = rows_to_check[rows_to_check['EchargedBattery'] <= 0]
    ebattery_issue = rows_to_check[rows_to_check['Ebattery'] <= rows_to_check['Ebattery_prev']]
    ebattr_issue = rows_to_check[rows_to_check['EbattR'] <= rows_to_check['EbattR_prev']]

    assert echarged_issue.empty, f"EchargedBattery not positive under charging conditions:\n{echarged_issue[['datetime', 'EchargedBattery']]}"
    assert ebattery_issue.empty, f"Ebattery did not increase under charging conditions:\n{ebattery_issue[['datetime', 'Ebattery_prev', 'Ebattery']]}"
    assert ebattr_issue.empty, f"EbattR did not increase under charging conditions:\n{ebattr_issue[['datetime', 'EbattR_prev', 'EbattR']]}"


def test_discharge_behavior(trip_data):
    """Test that when EchargedBattery < 0, Ebattery and EbattR decrease, EbattG stays the same."""
    if "noSM" in trip_data.name:
        pytest.skip("Skipping non-smart-charging trip data file.")

    df = trip_data.copy().sort_values('datetime').reset_index(drop=True)

    df['Ebattery_prev'] = df['Ebattery'].shift(1)
    df['EbattR_prev'] = df['EbattR'].shift(1)
    df['EbattG_prev'] = df['EbattG'].shift(1)

    discharging_rows = df[df['EchargedBattery'] < 0]

    if discharging_rows.empty:
        pytest.skip("No discharging events found in this trip file.")

    discharging_rows = discharging_rows.dropna(subset=['Ebattery_prev', 'EbattR_prev', 'EbattG_prev'])

    ebattery_issue = discharging_rows[discharging_rows['Ebattery'] >= discharging_rows['Ebattery_prev']]
    ebattr_issue = discharging_rows[discharging_rows['EbattR'] >= discharging_rows['EbattR_prev']]
    ebattg_issue = discharging_rows[discharging_rows['EbattG'] != discharging_rows['EbattG_prev']]

    assert ebattery_issue.empty, f"Ebattery did not decrease during discharging:\n{ebattery_issue[['datetime', 'Ebattery_prev', 'Ebattery']]}"
    assert ebattr_issue.empty, f"EbattR did not decrease during discharging:\n{ebattr_issue[['datetime', 'EbattR_prev', 'EbattR']]}"
    assert ebattg_issue.empty, f"EbattG changed during discharging (should stay constant):\n{ebattg_issue[['datetime', 'EbattG_prev', 'EbattG']]}"
