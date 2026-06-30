# src/data_processing.py
import numpy as np
import pandas as pd

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset and applies defensive data-cleaning checks:
    Handling missing values, duplicates, and geographic bounding limits.
    """
    df = pd.read_csv(filepath)
    
    # 1. Check for missing values in critical columns
    critical_cols = ['name', 'latitude', 'longitude']
    if df[critical_cols].isnull().any().any():
        df = df.dropna(subset=critical_cols)
        
    # 2. Remove exact duplicate coordinates if they exist
    df = df.drop_duplicates(subset=['latitude', 'longitude'])
    
    # 3. Apply corrected regional bounding box for Bosnia & Herzegovina
    LAT_MIN, LAT_MAX = 42.5, 45.3
    LON_MIN, LON_MAX = 15.7, 19.8  # 19.8 ensures extreme eastern points pass safely
    
    # Filter for coordinates that fall outside the boundary
    invalid_rows = df[
        (df['latitude'] < LAT_MIN) | (df['latitude'] > LAT_MAX) |
        (df['longitude'] < LON_MIN) | (df['longitude'] > LON_MAX)
    ]
    
    if not invalid_rows.empty:
        raise ValueError(
            f"Data Quality Alert: Found coordinates completely outside BiH borders:\n"
            f"{invalid_rows[['name', 'latitude', 'longitude']]}"
        )
        
    return df

def get_radial_coordinates(df: pd.DataFrame) -> np.ndarray:
    """
    Isolates only the latitude and longitude columns and converts them 
    to radians to fulfill the mathematical requirements of the Haversine metric.
    """
    coords_deg = df[['latitude', 'longitude']].values
    return np.radians(coords_deg)