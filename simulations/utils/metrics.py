import pandas as pd
import os


def compute_selfconsumption(ev, file_path):
    df = ev.trips.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Calculate the change in EbattR over time
    df['delta_EbattR'] = df['EbattR'].diff().fillna(0)

    # Only consider positive changes (charging, not discharging)
    df['delta_EbattR'] = df['delta_EbattR'].apply(lambda x: x if x > 0 else 0)

    # Initialize a list to store the monthly results
    results = []

    # Loop through each month (1 to 12)
    for month in range(1, 13):
        # Filter data for the current month
        monthly_df = df[df['datetime'].dt.month == month]

        # Skip if there are no entries for the month
        if monthly_df.empty:
            continue

        # Calculate total consumption for home and workplace
        home_cons = monthly_df[monthly_df['state'] == 'home']['delta_EbattR'].sum()
        workplace_cons = monthly_df[monthly_df['state'] == 'workplace']['delta_EbattR'].sum()

        # Store the result for the current month
        results.append({
            'ev_name': ev.name,
            'smart_charging': ev.smart,
            'month': month,
            'workplace': workplace_cons,
            'home': home_cons
        })

    # Convert the results into a DataFrame
    result_df = pd.DataFrame(results)

    # Write to CSV, appending if the file exists
    file_exists = os.path.exists(file_path)
    result_df.to_csv(file_path, mode='a', header=not file_exists, index=False)


def compute_from_grid(ev, file_path):
    # Copy the DataFrames
    df_home = ev.to_home.copy()
    df_workplace = ev.to_workplace.copy()
    df_workplace['datetime'] = pd.to_datetime(df_workplace['datetime'], errors='coerce')
    df_home['datetime'] = pd.to_datetime(df_home['datetime'], errors='coerce')

    df_home['total'] = df_home['energy'] + df_home['from_ev']
    df_workplace['total'] = df_workplace['energy'] + df_workplace['from_ev']

    results = []

    # Loop through each month (1 to 12)
    for month in range(1, 13):
        # Filter data for the current month for both home and workplace
        home_month_df = df_home[df_home['datetime'].dt.month == month]
        workplace_month_df = df_workplace[df_workplace['datetime'].dt.month == month]

        if home_month_df.empty and workplace_month_df.empty:
            print(f"No data for EV {ev.name} in month {month}. Skipping.")
            continue

        grid_power_home = home_month_df[home_month_df['total'] < 0]['total'].sum()
        grid_power_workplace = workplace_month_df[workplace_month_df['total'] < 0]['total'].sum()

        ev_power_home = home_month_df[home_month_df['from_ev'] > 0]['from_ev'].sum()
        ev_power_workplace = workplace_month_df[workplace_month_df['from_ev'] > 0]['from_ev'].sum()

        # Store the result for the current month
        results.append({
            'ev_name': ev.name,
            'smart_charging': ev.smart,
            'month': month,
            'grid_home': grid_power_home,
            'grid_workplace': grid_power_workplace,
            'ev_home': ev_power_home,
            'ev_workplace': ev_power_workplace
        })

    # Convert the results into a DataFrame
    result_df = pd.DataFrame(results)

    # Write to CSV, appending if file already exists
    file_exists = os.path.exists(file_path)
    result_df.to_csv(file_path, mode='a', header=not file_exists, index=False)


def compute_community_transfer(ev, file_path):
    trips = ev.trips.copy()
    home_df = ev.to_home.copy()
    workplace_df = ev.to_workplace.copy()

    # Ensure datetime columns are in datetime format
    trips['datetime'] = pd.to_datetime(trips['datetime'])
    home_df['datetime'] = pd.to_datetime(home_df['datetime'])
    workplace_df['datetime'] = pd.to_datetime(workplace_df['datetime'])

    # Add a month column
    trips['month'] = trips['datetime'].dt.month
    home_df['month'] = home_df['datetime'].dt.month
    workplace_df['month'] = workplace_df['datetime'].dt.month

    results = []

    for month in range(1, 13):
        monthly_trips = trips[trips['month'] == month]
        monthly_home = home_df[home_df['month'] == month]
        monthly_workplace = workplace_df[workplace_df['month'] == month]

        transferred_hw = 0  # home → workplace
        transferred_wh = 0  # workplace → home

        for i in range(len(monthly_trips) - 1):
            row = monthly_trips.iloc[i]
            next_row = monthly_trips.iloc[i + 1]

            # Ensure direction is home → workplace
            if row['state'] == 'home' and next_row['state'] == 'workplace':
                # Look for discharges before and after the trip
                discharge_home = monthly_home[
                    (monthly_home['datetime'] <= row['datetime']) & (monthly_home['from_ev'] > 0)
                    ]
                discharge_work = monthly_workplace[
                    (monthly_workplace['datetime'] >= next_row['datetime']) & (monthly_workplace['from_ev'] > 0)
                    ]

                if not discharge_home.empty and not discharge_work.empty:
                    e_home = discharge_home['from_ev'].iloc[-1]
                    e_work = discharge_work['from_ev'].iloc[0]
                    transferred_hw += min(e_home, e_work)

            # Now do workplace → home
            elif row['state'] == 'workplace' and next_row['state'] == 'home':
                discharge_work = monthly_workplace[
                    (monthly_workplace['datetime'] <= row['datetime']) & (monthly_workplace['from_ev'] > 0)
                    ]
                discharge_home = monthly_home[
                    (monthly_home['datetime'] >= next_row['datetime']) & (monthly_home['from_ev'] > 0)
                    ]

                if not discharge_work.empty and not discharge_home.empty:
                    e_work = discharge_work['from_ev'].iloc[-1]
                    e_home = discharge_home['from_ev'].iloc[0]
                    transferred_wh += min(e_work, e_home)

        results.append({
            'ev_name': ev.name,
            'smart_charging': ev.smart,
            'month': month,
            'home_to_workplace': transferred_hw,
            'workplace_to_home': transferred_wh
        })

    result_df = pd.DataFrame(results)

    file_exists = os.path.exists(file_path)
    result_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

