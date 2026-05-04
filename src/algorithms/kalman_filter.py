import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class KalmanTrack:
    state: np.ndarray              # [x, y, z, vx, vy, vz, q/p]
    covariance: np.ndarray         # 7x7 uncertainty matrix
    chi2: float = 0.0              # Sum of chi-squared contributions
    ndf: int = 0                   # Degrees of freedom
    hits_used: int = 0             # Number of hits processed


class KalmanFilter:
    
    def __init__(self, process_noise: float = 0.001,
                 magnetic_field: np.ndarray = None):
    
        self.process_noise = process_noise
        self.magnetic_field = magnetic_field or np.array([0, 0, 5.0])
    
    def initialize_track(self, hits: List) -> KalmanTrack:
    
        if len(hits) < 2:
            return None
        
        # Extract first few hit positions
        positions = np.array([[h.x, h.y, h.z] for h in hits[:min(3, len(hits))]])
        
        # Estimate direction
        direction = positions[1] - positions[0]
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 1e-10:
            direction = direction / direction_norm
        else:
            direction = np.array([1, 0, 0])
        
        # Initial state
        state = np.array([
            float(hits[0].x),
            float(hits[0].y),
            float(hits[0].z),
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
            0.1  # q/p ratio
        ])
        
        # Initial covariance (uncertainty)
        covariance = np.eye(7, dtype=float)
        
        # Position: reasonably confident (from 2-3 hits average)
        covariance[0, 0] = 0.001      # 0.03 cm std dev
        covariance[1, 1] = 0.001
        covariance[2, 2] = 0.001
        
        # Direction: less confident
        covariance[3, 3] = 0.01
        covariance[4, 4] = 0.01
        covariance[5, 5] = 0.01
        
        # q/p: quite uncertain
        covariance[6, 6] = 1.0
        
        return KalmanTrack(state=state, covariance=covariance)
    
    def predict(self, track: KalmanTrack, distance: float):
    
        state = track.state.copy()
        
        # Current position and direction
        position = state[0:3]
        direction = state[3:6]
        
        # Normalize direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-10:
            direction = direction / direction_norm
            state[3:6] = direction
        
        # Predict new position
        state[0:3] = position + distance * direction
        track.state = state
        
        # Covariance grows (uncertainty increases)
        # Position uncertainty grows quadratically with distance
        position_growth = self.process_noise * (distance ** 2)
        track.covariance[0, 0] += position_growth
        track.covariance[1, 1] += position_growth
        track.covariance[2, 2] += position_growth
        
        # Direction uncertainty grows linearly
        direction_growth = self.process_noise * distance
        track.covariance[3, 3] += direction_growth
        track.covariance[4, 4] += direction_growth
        track.covariance[5, 5] += direction_growth
    
    def update(self, track: KalmanTrack, hit) -> float:
    
        
        # Measurement matrix: measure [x, y, z]
        H = np.zeros((3, 7))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        
        # Measurement noise (MUST match ParticleGenerator!)
        R = np.diag([hit.error_x**2, hit.error_y**2, hit.error_z**2])
        
        # Measurement
        z = np.array([hit.x, hit.y, hit.z])
        
        # Predicted measurement
        z_pred = H @ track.state
        
        # Innovation (residual)
        innovation = z - z_pred
        
        # Innovation covariance
        S = H @ track.covariance @ H.T + R
        
        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
            K = track.covariance @ H.T @ S_inv
        except np.linalg.LinAlgError:
            K = np.zeros((7, 3))
            S_inv = np.zeros((3, 3))
        
        # Update state
        track.state = track.state + K @ innovation
        
        # Update covariance
        I = np.eye(7)
        track.covariance = (I - K @ H) @ track.covariance
        
        # Chi-squared contribution
        try:
            chi2 = float(innovation @ S_inv @ innovation)
        except:
            chi2 = 0.0
        
        track.chi2 += chi2
        track.ndf += 1
        track.hits_used += 1
        
        return chi2
    
    def fit_track(self, hits: List) -> KalmanTrack:
    
        if len(hits) < 2:
            return None
        
        # Initialize
        track = self.initialize_track(hits)
        if track is None:
            return None
        
        # Process each hit
        for i in range(1, len(hits)):
            distance = 1.0  # 1.0 cm spacing between layers
            
            # Predict
            self.predict(track, distance)
            
            # Update
            self.update(track, hits[i])
        
        return track

