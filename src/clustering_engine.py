# src/clustering_engine.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def calculate_k_distances(coordinates_rad: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Calculates the great-circle distance of each point to its k-th nearest neighbor.
    Used to plot the K-distance graph to determine the optimal epsilon value.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric='haversine').fit(coordinates_rad)
    distances, _ = nbrs.kneighbors(coordinates_rad)
    
    # Sort by the distance to the k-th nearest neighbor (column index k-1)
    sorted_k_distances = np.sort(distances[:, k-1])
    return sorted_k_distances

def run_spatial_dbscan(coordinates_rad: np.ndarray, eps_km: float, min_samples: int) -> DBSCAN:
    """
    Executes DBSCAN using the Haversine metric. Automatically translates 
    the kilometer epsilon into radians for calculation.
    """
    EARTH_RADIUS_KM = 6371.009
    
    # Mathematical conversion: epsilon in kilometers transformed to radians
    eps_rad = eps_km / EARTH_RADIUS_KM
    
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    db.fit(coordinates_rad)
    return db
