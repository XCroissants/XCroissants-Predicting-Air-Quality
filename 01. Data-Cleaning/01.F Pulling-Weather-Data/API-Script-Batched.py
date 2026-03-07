#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batched Open-Meteo historical weather fetcher for Île-de-France communes.
Splits 1,287 locations into batches of 25 to avoid HTTP 414 (URI Too Long) errors.
Results are saved to weather_2017_idf.csv after EACH batch so progress is never lost.

To resume from a specific batch after a failure, change START_BATCH below.
For example, if batch 14 failed, set START_BATCH = 13 (0-indexed).
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import time
import os

# ── Setup the Open-Meteo API client with cache and retry on error ──────────────
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ── Load communes from CSV ────────────────────────────────────────────────────
communes   = pd.read_csv("/Users/antonioraphael/Documents/PROJECT-CLONES/Data-Storage/AirQualityData/Clean_Data/Commune_Centroid.csv")
latitudes  = communes["Lat"].tolist()
longitudes = communes["Lon"].tolist()
insee_codes = communes["insee"].tolist()

url = "https://archive-api.open-meteo.com/v1/archive"

BATCH_SIZE  = 25
START_BATCH = 0       # Change this to resume from a specific batch (0-indexed)
OUTPUT_CSV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_2017_idf.csv")

# Helper: fetch one batch and return a list of DataFrames
def fetch_batch(lats, lons, codes):
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": "2017-01-01",
        "end_date": "2017-12-31",
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "daylight_duration",
            "sunshine_duration",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
        ],
    }

    responses = openmeteo.weather_api(url, params=params)
    batch_frames = []

    for i, response in enumerate(responses):
        daily = response.Daily()
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ),
            "insee":     codes[i],
            "latitude":  response.Latitude(),
            "longitude": response.Longitude(),
            "temperature_2m_mean":          daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_max":           daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_min":           daily.Variables(2).ValuesAsNumpy(),
            "daylight_duration":            daily.Variables(3).ValuesAsNumpy(),
            "sunshine_duration":            daily.Variables(4).ValuesAsNumpy(),
            "precipitation_sum":            daily.Variables(5).ValuesAsNumpy(),
            "rain_sum":                     daily.Variables(6).ValuesAsNumpy(),
            "snowfall_sum":                 daily.Variables(7).ValuesAsNumpy(),
            "wind_speed_10m_max":           daily.Variables(8).ValuesAsNumpy(),
            "wind_gusts_10m_max":           daily.Variables(9).ValuesAsNumpy(),
            "wind_direction_10m_dominant":  daily.Variables(10).ValuesAsNumpy(),
        }
        batch_frames.append(pd.DataFrame(data=daily_data))

    return batch_frames

# Main loop
total       = len(latitudes)
num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Loaded {total} communes from CSV.")
print(f"Output will be saved to: {OUTPUT_CSV}")
print(f"Starting from batch {START_BATCH + 1}/{num_batches}\n")

for b in range(START_BATCH, num_batches):
    start       = b * BATCH_SIZE
    end         = min(start + BATCH_SIZE, total)
    batch_lats  = latitudes[start:end]
    batch_lons  = longitudes[start:end]
    batch_codes = insee_codes[start:end]

    print(f"Fetching batch {b+1}/{num_batches} (locations {start+1}-{end})...")
    try:
        frames   = fetch_batch(batch_lats, batch_lons, batch_codes)
        batch_df = pd.concat(frames, ignore_index=True)
        batch_df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
        print(f"  Batch {b+1} saved ({len(batch_df):,} rows written to {OUTPUT_CSV})")

    except Exception as e:
        if "request limit exceeded" in str(e).lower():
            print(f"  Rate limit hit - waiting 90 seconds before retrying...")
            time.sleep(90)
            try:
                frames   = fetch_batch(batch_lats, batch_lons, batch_codes)
                batch_df = pd.concat(frames, ignore_index=True)
                batch_df.to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                print(f"  Batch {b+1} saved after retry ({len(batch_df):,} rows written)")
            except Exception as e2:
                print(f"  Batch {b+1} failed again: {e2}")
                print(f"  To resume later, set START_BATCH = {b} at the top of the script.")
        else:
            print(f"  Batch {b+1} failed: {e}")
            print(f"  To resume later, set START_BATCH = {b} at the top of the script.")

    if b < num_batches - 1:
        time.sleep(30)

print(f"\nAll done! Data saved to {OUTPUT_CSV}")
